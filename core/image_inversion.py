"""This lib curates functions to invert images for modern GANs, esp. functions to invert specific GANs: 
* Projection + Adam, fit Euclidean transform together with the latent code.
* BasinCMA, Use hybrid of CMAES and Adam to search for a good starting point at first and then use Adam to optimize from there.

These has been tested to invert ProgGrowing GAN, StyleGAN2, BigGAN. 
Note Hessian preconditioning is supported for all these methods. 

Binxu Wang
2020.8-2021.6
"""
import os
import cma
import tqdm
from os.path import join
import torch, numpy as np
from scipy.linalg import block_diag
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision.transforms import Resize, ToTensor
import matplotlib.pylab as plt
from skimage.transform import rescale, resize
# from GAN_utils import StyleGAN2_wrapper, loadStyleGAN2
from pytorch_pretrained_biggan import truncated_noise_sample, one_hot_from_int
from lpips import LPIPS
from .load_hessian_data import load_Haverage
from .torch_utils import show_imgrid, save_imgrid, ToPILImage, make_grid
def MSE(im1, im2, mask=None):
    """Distance function between images. consider loss weighted by mask if available
    Inputs:
        im1, im2: torch tensor images of shape [B, C, H, W] and [1, C, H, W] or vice verse
        mask: torch tensor of size [sampn, H, W] or as im1, im2. dtype can be Bool or float.

    Output:
        distvec: Distance between image in im1 batch to im2 or vice versa. a torch tensor of shape [B,]
    """
    if mask is None:
        return (im1 - im2).pow(2).mean(dim=[1,2,3])
    else:
        valnum = mask.sum([1, 2])
        diffsum = ((im1 - im2).pow(2).mean(1) * mask).sum([1, 2])
        return diffsum / valnum

D = LPIPS(net="squeeze")#"vgg"
D.cuda()
D.requires_grad_(False)
D.spatial = True
def mask_LPIPS(im1, im2, mask=None, D=D):
    """Distance function between images. Using LPIPS distance `D` to create distance spatial map.
        consider loss weighted by mask if available
    Inputs:
        im1, im2: torch tensor images of shape [B, C, H, W] and [1, C, H, W] or vice verse
            Not sure if [B, C, H, W] and [B, C, H, W] work.
        mask: torch tensor of size [sampn, H, W] or as im1, im2. dtype can be Bool or float.
        D: an LPIPS distance function, set `spatial` to True. 
            for example `D = LPIPS(net="squeeze").cuda();D.requires_grad_(False);D.spatial = True`

    Output:
        distvec: a torch tensor of shape [B,]
    """
    diffmap = D(im1, im2)  # note there is a singleton channel dimension
    if mask is None:
        return diffmap.mean([1, 2, 3])
    else:
        diffsum = (diffmap[:, 0, :, :] * mask).sum([1, 2])
        valnum = mask.sum([1, 2])
        return diffsum / valnum

# Image preprocess, crop and resized
def crop_rsz(img, crop_param="center", rsz_param=(256,256)):
    """
    img: np array image 
    crop_param: "center", "none" or a length 4 array, list or tuple. 
    rsz_param: tuple to send in `resize` function. 
    """
    if img.ndim == 3:
        H, W, C = img.shape
        if C == 4:
            img = img[:, :, :3]
    elif img.ndim == 2:
        H, W = img.shape
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
    else:
        raise ValueError("Check shape of img input")
    if crop_param == "center":
        if H >= W:
            marg = (H - W)//2
            window = (0, marg, W, W)
        else:
            marg = (W - H) // 2
            window = (marg, 0, H, H)
    elif crop_param == "none":
        window = (0, 0, W, H)
    else:
        window = crop_param
    cropped = img[window[1]:window[1]+window[3], window[0]:window[0]+window[2], :]
    rsz_cropped = resize(cropped, rsz_param)
    return rsz_cropped

# Define Basis (Lie Algebra) for rotation matrix SO(2)
alpha_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).cuda()
beta_mat = torch.tensor([[0.0, 1.0], [-1.0, 0.0]]).cuda()
def img_project(srctsr_rsz, G, D, wspace=False, stepn=300, sampN=3,
                initvec=None, hess_precon=True, evctsr=None,
                euclid_tfm=True, tfm_target=False, initEuclid=None, regist_freq=10,
                imgnm="img", paramstr="", RND=None, figdir="", ):
    """ Fitting the image together with an euclidean transform matrix. 
    This "general-purpose" inversion program has been used for PGGAN and StyleGAN. 
    BigGAN has its own inversion algorithm below due to its unique latent space structure.

    Inputs: 
        srctsr_rsz: torch tensor of target image, shape [1, 3, H, W]. 
        G: Generator equipped with `visualize` and `sample_vector` function. 
            PGGAN and StyleGAN2 have been tested.
        D: function that takes 2 image tensors and compute image distance.
            Can take optional 3rd parameter, the spatial mask to compute distance.
            e.g. mask_LPIPS or MSE 
        
        initvec: A given initial vector. If None, then sample one from the given GAN.
            If not it could be a proper length np array or torch tensor, it will be copied to initialize 
            fit. 
    
    Optim Settings
        sampN: How many threads to invert in parallel. Int.
        stepn: Adam step numbers in total. Int.
    
    Hessian Preconditioning: 
        hess_precon: whether or not using hessian preconidtioning. Bool.
        evctsr: eigenvector tensor to rotation the hidden space. Default to be None.
            Required if hess_precon is True. if given, it's a Torch Tensor shape (n, n)
    
    Euclidean Transform Parameters: 
        euclid_tfm: Do euclidean transform to target or source. Bool.
        tfm_target:  transform each individual image instead of target. Bool.
        initEuclid: Initial Euclidean parameters. torch tensor of size (4, ). Torch Tensor.
        regist_freq: Frequency (how many latent update steps in between) for updating euclidean transform parameters. Int
    
    Output Parameters:    
        imgnm: 
        paramstr: 
        RND: 
        figdir: 

    Demo: 
        srcimg = plt.imread(join(imroot, imgnm))
        srctsr = ToTensor()(srcimg).unsqueeze(0)
        imgtsr_rsz = F.interpolate(srctsr, [256, 256]).cuda()
        RND = np.random.randint(1000)
        # No Hessian preconditioning with euclidean transform
        refvecs, refimgs, EuclidParam, mse_col, dsim_col = img_project(imgtsr_rsz, G, mask_LPIPS, imgnm=imgnm,
                       stepn=250, sampN=4, initvec=None, hess_precon=False, evctsr=None,
                       initEuclid=None, regist_freq=10, euclid_tfm=True, tfm_target=False, 
                       RND=RND, paramstr="Z_noH_Eucl", figdir=figdir)
        # If you want to use Hessian matrix to do pre-condition 
        H, eva, evc = load_Haverage("StyleGAN2-Face512_Z", descend=True)
        evctsr = torch.tensor(evc).cuda().float()
        # Hessian preconditioning with euclidean transform (for fit image)
        refvecs, refimgs, EuclidParam, mse_col, dsim_col = img_project(imgtsr_rsz, G, mask_LPIPS, imgnm=imgnm,
                       stepn=250, sampN=4, initvec=None, hess_precon=True, evctsr=evctsr,
                       initEuclid=None, regist_freq=10, euclid_tfm=True, tfm_target=False, 
                       RND=RND, paramstr="Z_H_Eucl", figdir=figdir)
        # Hessian preconditioning with euclidean transform (for target)
        refvecs, refimgs, EuclidParam, mse_col, dsim_col = img_project(imgtsr_rsz, G, mask_LPIPS, imgnm=imgnm,
                       stepn=250, sampN=4, initvec=None, hess_precon=True, evctsr=evctsr,
                       initEuclid=None, regist_freq=10, euclid_tfm=True, tfm_target=True, 
                       RND=RND, paramstr="Z_H_Eucl", figdir=figdir)
        # Hessian preconditioning with no euclidean transform
        refvecs, refimgs, EuclidParam, mse_col, dsim_col = img_project(imgtsr_rsz, G, mask_LPIPS, imgnm=imgnm,
                       stepn=250, sampN=4, initvec=None, hess_precon=True, evctsr=evctsr,
                       euclid_tfm=False, RND=RND, paramstr="Z_H", figdir=figdir)

    Obsolete: OK so the Euclid tfm Part is not working properly... need to fix it
    """
    # if wspace:
    #     G.use_wspace(True)
    #     H, eva, evc = load_Haverage("StyleGAN2-Face512_W", descend=True)
    #     evctsr = torch.tensor(evc).cuda().float()
    # else:
    #     G.use_wspace(False)
    #     H, eva, evc = load_Haverage("StyleGAN2-Face512_Z", descend=True)
    #     evctsr = torch.tensor(evc).cuda().float()
    if wspace:
        raise NotImplementedError("Use the W space for style GAN but not implemented yet.")
    srctsr_rsz = srctsr_rsz.float()
    preconMat = evctsr if hess_precon else torch.eye(evctsr.shape[1]).cuda()
    if initvec is None:
        fitvec = G.sample_vector(sampN, device='cuda')
    else:
        fitvec = torch.tensor(initvec).detach().clone().cuda()
    fitcoef = fitvec @ preconMat
    fitcoef.requires_grad_(True)
    optimizer = Adam([fitcoef, ], lr=0.05, weight_decay=1E-4)  # 0.01 is good step for StyleGAN2
    if initEuclid is None:
        EuclidParam = torch.tensor([1.0, 0.0, 0.0, 0.0]).cuda().float()  # 1.0,
    else:
        EuclidParam = torch.tensor(initEuclid).cuda().float()  # 1.0,
    if not tfm_target:  # transform each individual image instead of target.
        EuclidParam = EuclidParam.repeat(sampN, 1)
    if euclid_tfm:
        EuclidParam.requires_grad_(True)
        regist_optim = SGD([EuclidParam, ], lr=0.003, )  # 0.01 is good step for StyleGAN2
    # SGD 0.05-0.01 is not good
    dsim_col = []
    mse_col = []
    for step in range(stepn):
        optimizer.zero_grad()
        if euclid_tfm:  regist_optim.zero_grad()
        fitvec = fitcoef @ preconMat.T
        fittsr = G.visualize(fitvec)
        if euclid_tfm:  # Euclid Transform before computing loss. 
            if tfm_target: # Transform target
                theta = torch.cat((EuclidParam[0]*
                   (alpha_mat * torch.cos(EuclidParam[1]) + beta_mat * torch.sin(EuclidParam[1])),
                               EuclidParam[2:].unsqueeze(1)), dim=1).unsqueeze(0)
                grid = F.affine_grid(theta, srctsr_rsz.size())
                validmsk = (grid[:, :, :, 0] > -1) * (grid[:, :, :, 0] < 1) * \
                           (grid[:, :, :, 1] > -1) * (grid[:, :, :, 1] < 1)
                srctsr_rsz_tfm = F.grid_sample(srctsr_rsz, grid)
                dsim = D(srctsr_rsz_tfm, fittsr, validmsk)
                MSE_err = MSE(srctsr_rsz_tfm, fittsr, validmsk)
            else: # Transform each and every source. 
                # Scale * Rotation * Translation
                theta = torch.cat(tuple(
                    torch.cat((EuclidParam[i, 0]*
                        (alpha_mat * torch.cos(EuclidParam[i, 1]) + beta_mat * torch.sin(EuclidParam[i, 1])),
                        EuclidParam[i, 2:].unsqueeze(1)), dim=1).unsqueeze(0)
                    for i in range(sampN)))
                grid = F.affine_grid(theta, fittsr.size())
                fittsr_tfm = F.grid_sample(fittsr, grid)
                validmsk = (grid[:, :, :, 0] > -1) * (grid[:, :, :, 0] < 1) * \
                           (grid[:, :, :, 1] > -1) * (grid[:, :, :, 1] < 1)
                dsim = D(srctsr_rsz, fittsr_tfm, validmsk)
                MSE_err = MSE(srctsr_rsz, fittsr_tfm, validmsk)
        else: # without euclidean transform 
            dsim = D(srctsr_rsz, fittsr)
            MSE_err = MSE(srctsr_rsz, fittsr)
        loss = dsim + MSE_err
        dsim_col.append(dsim.detach().cpu().numpy())
        mse_col.append(MSE_err.detach().cpu().numpy())
        loss.sum().backward()
        optimizer.step()
        if euclid_tfm and (step) % regist_freq:  # register image at a lower rate.
            regist_optim.step()
            for i in range(10): # Optimizing the positioning without changing the code. 
                regist_optim.zero_grad()
                if tfm_target:
                    theta = torch.cat((EuclidParam[0]*
                       (alpha_mat * torch.cos(EuclidParam[1]) + beta_mat * torch.sin(EuclidParam[1])),
                                   EuclidParam[2:].unsqueeze(1)), dim=1).unsqueeze(0)
                    grid = F.affine_grid(theta, srctsr_rsz.size())
                    srctsr_rsz_tfm = F.grid_sample(srctsr_rsz, grid)
                    dsim = D(srctsr_rsz_tfm, fittsr)
                    MSE_err = MSE(srctsr_rsz_tfm, fittsr)
                else:
                    theta = torch.cat(tuple(
                        torch.cat((EuclidParam[i, 0]*
                            (alpha_mat * torch.cos(EuclidParam[i, 1]) + beta_mat * torch.sin(EuclidParam[i, 1])),
                            EuclidParam[i, 2:].unsqueeze(1)), dim=1).unsqueeze(0)
                        for i in range(sampN)))
                    grid = F.affine_grid(theta, fittsr.size())
                    fittsr_tfm = F.grid_sample(fittsr, grid)
                    dsim = D(srctsr_rsz, fittsr_tfm)
                    MSE_err = MSE(srctsr_rsz, fittsr_tfm)
                regist_optim.step()
        if step % 10 == 0:
            print(
                "LPIPS %.3f MSE %.3f norm %.1f" % (dsim.min().item(), MSE_err.min().item(), fitvec.norm(dim=1).mean()))
    if RND is None: RND = np.random.randint(1000)
    if not euclid_tfm:
        im_res = show_imgrid([srctsr_rsz, fittsr.detach()], )
    elif tfm_target:
        im_res = show_imgrid([srctsr_rsz, srctsr_rsz_tfm.detach(), fittsr.detach()], )
    else:
        im_res = show_imgrid([srctsr_rsz, fittsr_tfm.detach()], )
    im_res.save(join(figdir, "%s_%s%04d.png" % (imgnm, paramstr, RND)))
    refvecs = fitvec.detach().clone()
    refimgs = fittsr.detach().clone()
    mse_col = np.array(mse_col)
    dsim_col = np.array(dsim_col)
    torch.save({"fitvecs":refvecs, "fitimgs":refimgs, "EuclidParams":EuclidParam.detach(), "dsim_col":dsim_col, "mse_col":mse_col,
                "initvec":initvec, "initEuclid":initEuclid},
            join(figdir, "%s_%s%04d.pt" % (imgnm, paramstr, RND)))
    # Visualize optim trajectory
    plt.figure()
    plt.plot(mse_col, label="MSE")
    plt.plot(dsim_col, label="LPIPS")
    plt.ylabel("error")
    plt.xlabel("step num")
    plt.title("%s optim %s %d"%(imgnm, paramstr, RND))
    plt.savefig(join(figdir, "%s_%s%04d_score_traj.png" % (imgnm, paramstr, RND)))
    return refvecs, refimgs, EuclidParam, mse_col, dsim_col


def BasinCMA_BigGAN(target_tsr, G, ImDist, cmasteps=30, gradsteps=40, finalgrad=500, batch_size=4, 
             basis="none", basisvec=torch.eye(256).cuda(), CMApostAdam=False, RND=None, savedir="", imgnm="",
             classvec_init=None, fixclassvec=False, L2penalty=None, classpenalty=None):
    """ BasinCMA (CMA Adam hybrid optimization) for image inversion esp. for BigGAN
    This version is adapted for BigGAN for its separated Class and Noise space. 
    Inputs:
        target_tsr: torch tensor of an image shape like (3, 256, 256) or (1, 3, 256, 256)
        G: Wrapped generator 
        ImDist: lpips or other image distance function
        basis: str can one from "none", "all" or "sep". 
        basisvec: a torch tensor of shape (256, 256)

    BasinCMA parameters: 
        cmasteps: CMA steps for initial point optimization . Int
        gradsteps: Gradient steps after each seeds in CMA step . Int
        batch_size: How many initial points found by CMA are optimized in gradsteps together? This batch requires grad through GAN so cannot be too large.
            if gradsteps==0, then this parameter is disabled. . Int
        finalgrad: Final Adam step after optimization of initial values.  . Int
        CMApostAdam: Use the Adam optimized codes to feedback to CMA algorithm. If False use the initial codes to feedback. 
        
    Save parameters: 
        RND: random integer identifier for multiple trials.
        savedir: 
        imgnm: 
    
    Demo Usage: 
        srcimg = plt.imread(join(imroot, imgnm))
        srcimg_rsz = crop_rsz(srcimg, crop_param="center", )
        imgtsr_rsz = ToTensor()(srcimg_rsz).unsqueeze(0).float().cuda()
        # No CMA pure gradient version
        imgs_final, codes_final, scores_final, L1score_final, Record = BasinCMA_BigGAN(imgtsr_rsz, G, ImDist,
           cmasteps=0, gradsteps=0, finalgrad=600, batch_size=4, basis="all", basisvec=basisdict["all"],
           CMApostAdam=False, savedir=saveroot, imgnm=imgnm)

        # No gradient pure CMA version
        imgs_final, codes_final, scores_final, L1score_final, Record = BasinCMA_BigGAN(imgtsr_rsz, G, ImDist,
           cmasteps=100, gradsteps=0, finalgrad=0, batch_size=25, basis="all", basisvec=basisdict["all"],
           CMApostAdam=False, savedir=saveroot, imgnm=imgnm+"_cma_")
        
        # CMA + Adam version (larger batch size could be used.)
        imgs_final, codes_final, scores_final, L1score_final, Record = BasinCMA_BigGAN(srctsr_rsz, G, ImDist,
               cmasteps=80, gradsteps=0, finalgrad=400, batch_size=30, basis="all", basisvec=basisdict["all"],
               CMApostAdam=False, savedir=saveroot, imgnm=imgnm)

        # hybrid, BasinCMA + Adam version
        imgs_final, codes_final, scores_final, L1score_final, Record = BasinCMA_BigGAN(imgtsr_rsz, G, ImDist,
           cmasteps=10, gradsteps=10, finalgrad=600, batch_size=4, basis="all", basisvec=basisdict["all"],
           CMApostAdam=False, savedir=saveroot, imgnm=imgnm)

        # Add penalty to keep the search around the target class. 
        imgs_final, codes_final, scores_final, L1score_final, Record = BasinCMA_BigGAN(srctsr_rsz, G, ImDist,
               cmasteps=80, gradsteps=0, finalgrad=400, batch_size=30, basis="all", basisvec=basisdict["all"],
               CMApostAdam=False, savedir=saveroot, imgnm=imgnm+"_cma_pen", classvec_init=monkey_vec,
               L2penalty=(0.09, 0), classpenalty=0.4)
    """
    # choose the basis vector to use in Adam optimization
    # basisvec = {"all": evc_all, "sep": evc_sep, "none": evc_none}[basis]
    if target_tsr.ndim == 3:
        target_tsr = target_tsr.unsqueeze(0)
    Record = {"L1cma": [],"dsimcma": [], "L1Adam": [], "dsimAdam": [], "L1refine":[], "dsimrefine":[], "classnorm":[], "noisenorm":[]}
    if L2penalty is None:
        L2noise, L2class = 0, 0 
    elif isinstance(L2penalty, tuple):
        L2noise = L2penalty[0]
        L2class = L2penalty[1]
    else:
        L2noise = L2penalty
        L2class = L2penalty
    if classpenalty is None:
        class_dev_pen = 0
    else:
        class_dev_pen = classpenalty

    classnormref = 1.8
    noisenormref = 12
    fixnoise = truncated_noise_sample(1, 128)
    if classvec_init is None:
        classvec_init = np.zeros(128)
    else:
        assert len(classvec_init) == 128
    class_init_tsr = torch.tensor(classvec_init).unsqueeze(0).float().cuda()
    optim_noise = cma.CMAEvolutionStrategy(fixnoise, 0.4)#0.4)  # 0.2
    optim_class = cma.CMAEvolutionStrategy(classvec_init, 0.2)#0.12)  # 0.06
    # noise_vec = torch.from_numpy(fixnoise)
    RND = np.random.randint(1E6) if RND is None else RND
    # Part I: CMA optimization
    # Outer Loop: CMA optimization of initial points
    for i in tqdm.trange(cmasteps, desc="CMA steps"):
        class_codes = optim_class.ask()
        noise_codes = optim_noise.ask()
        # TODO: boundary handling by projection in code space
        # Evaluate the cma proposed codes `latent_code` at first.
        codes_tsr = torch.from_numpy(np.array(class_codes)).float()
        noise_tsr = torch.from_numpy(np.array(noise_codes)).float()
        latent_code = torch.cat((noise_tsr, codes_tsr), dim=1).cuda()  # this initialize inner loop
        with torch.no_grad():
            imgs = G.visualize(latent_code)
            dsims = ImDist(imgs, target_tsr).squeeze()
            L1dsim = (imgs - target_tsr).abs().mean([1, 2, 3])
        scores = dsims.detach().cpu().numpy()
        L1score = L1dsim.detach().cpu().numpy()
        class_dev = (codes_tsr - class_init_tsr.cpu()).norm(dim=1).numpy()
        classnorm = codes_tsr.norm(dim=1).numpy()
        noisenorm = noise_tsr.norm(dim=1).numpy()
        print("step %d pre-ADAM dsim %.3f L1 %.3f (class norm %.2f noise norm %.2f class dev %.2f)" % (
            i, scores.mean(), L1score.mean(), classnorm.mean(), noisenorm.mean(), class_dev.mean().item()))
        Record["L1cma"].append(L1score)
        Record["dsimcma"].append(scores)
        # Inner loop: ADAM optimization from the cma proposed codes `latent_code` batch by batch
        codes_post = np.zeros_like(np.hstack((noise_codes, class_codes)))
        scores_post = np.zeros_like(scores)
        L1score_post = np.zeros_like(L1score)
        if gradsteps > 0: # Hybrid Adam + CMA
            csr = 0
            while csr < len(class_codes):  # pbar = tqdm.tqdm(total=len(codes), initial=csr, desc="batchs")
                csr_end = min(len(class_codes), csr + batch_size)
                # codes_batch = codes_tsr[csr:csr_end, :].detach().clone().requires_grad_(True)
                coef_batch = (latent_code[csr:csr_end, :] @ basisvec).detach().clone().requires_grad_(True)
                optim = Adam([coef_batch], lr=0.05, )
                for step in range(gradsteps):  # tqdm.trange(gradsteps, desc="ADAM steps"):
                    optim.zero_grad()
                    latent_batch = coef_batch @ basisvec.T
                    imgs = G.visualize(latent_batch)
                    dsims = ImDist(imgs, target_tsr).squeeze()
                    L1dsim = (imgs - target_tsr).abs().mean([1, 2, 3])
                    loss = (dsims + L1dsim).sum()
                    loss.backward()
                    optim.step()
                    if (step + 1) % 10 == 0:
                        print("step %d dsim %.3f L1 %.3f" % (step, dsims.mean().item(), L1dsim.mean().item(),))
                code_batch = (coef_batch @ basisvec.T).detach().cpu().numpy() # Debug! @Apr. 1st Before there is a bug.
                scores_batch = dsims.detach().cpu().numpy()
                L1score_batch = L1dsim.detach().cpu().numpy()
                codes_post[csr:csr_end, :] = code_batch
                scores_post[csr:csr_end] = scores_batch
                L1score_post[csr:csr_end] = L1score_batch
                csr = csr_end
            # Use the ADAM optimized scores as utility for `latent_code` and do cma update
            print("step %d post-ADAM dsim %.3f L1 %.3f (norm %.2f, norm %.2f)" % (
                i, scores_post.mean(), L1score_post.mean(),
                np.linalg.norm(codes_post[:, 128:], axis=1).mean(),
                np.linalg.norm(codes_post[:, :128], axis=1).mean()
            ))
        else:  # if no grad step is performed
            scores_post = scores
            L1score_post = L1score
            codes_post = np.hstack((noise_codes, class_codes))
        Record["L1Adam"].append(L1score_post)
        Record["dsimAdam"].append(scores_post)
        cma_scores = scores_post + L1score_post
        if L2penalty is not None:
            cma_scores = cma_scores + L2class * np.maximum(classnorm-classnormref, 0) \
                    + L2noise * np.maximum(noisenorm-noisenormref, 0)
        if classpenalty is not None:
            cma_scores = cma_scores + class_dev_pen * class_dev

        if CMApostAdam:
            optim_class.tell(codes_post[:, :128], cma_scores)
            optim_noise.tell(codes_post[:, 128:], cma_scores)
        else:
            optim_class.tell(class_codes, cma_scores)
            optim_noise.tell(noise_codes, cma_scores)

    # Sort the scores and find the codes with the least scores to be final refined
    idx = np.argsort((L1score_post + scores_post))
    codes_batch = torch.from_numpy(codes_post[idx[:4]]).float().cuda()
    fit_img = G.visualize(codes_batch)
    CMAfitimg = ToPILImage()(make_grid(torch.cat((fit_img, target_tsr)).cpu()))
    CMAfitimg.save(join(savedir, "%s_CMA_final%06d.jpg" % (imgnm, RND)))
    CMAfitimg.show()

    # Part II: refinement of the codes using Adam from the initial points got in part I. 
    # Linear Reparametrization using basisvec
    coef_batch = (codes_batch @ basisvec).detach().clone().requires_grad_(True)
    optim = Adam([coef_batch], lr=0.05, )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=200, gamma=0.5)
    for step in range(finalgrad):  # tqdm.trange(gradsteps, desc="ADAM steps"):
        optim.zero_grad()
        # latent_code = torch.cat((noise_vec.repeat(codes_batch.shape[0], 1), codes_batch), dim=1).cuda()
        latent_code = coef_batch @ basisvec.T  #evc_all.T
        imgs = G.visualize(latent_code)
        dsims = ImDist(imgs, target_tsr).squeeze()
        L1dsim = (imgs - target_tsr).abs().mean([1, 2, 3])
        class_dev = (latent_code[:, 128:] - class_init_tsr).norm(dim=1)
        classnorm = latent_code[:, 128:].norm(dim=1)
        noisenorm = latent_code[:, :128].norm(dim=1)
        loss = dsims + L1dsim
        if L2penalty is not None:
            loss = loss + L2class * torch.clip(classnorm-classnormref, 0) \
                    + L2noise * torch.clip(noisenorm-noisenormref, 0)
        if classpenalty is not None:
            loss = loss + class_dev_pen * class_dev
        
        loss.sum().backward()
        optim.step()
        scheduler.step()
        Record["L1refine"].append(L1dsim.detach().cpu().numpy())
        Record["dsimrefine"].append(dsims.detach().cpu().numpy())
        Record["classnorm"].append(classnorm.detach().cpu().numpy())
        Record["noisenorm"].append(noisenorm.detach().cpu().numpy())
        if (step + 1) % 10 == 0:
            print("step %d dsim %.3f L1 %.3f (norm %.2f class_deviation %.2f)" % (
                step, dsims.mean().item(), L1dsim.mean().item(), latent_code.norm(dim=1).mean().item(),
                class_dev.mean().item()))
    scores_final = dsims.detach().cpu().numpy()
    L1score_final = L1dsim.detach().cpu().numpy()
    finalimg = ToPILImage()(make_grid(torch.cat((imgs, target_tsr)).cpu()))
    finalimg.save(join(savedir, "%srefinefinal%06d.jpg" % (imgnm, RND)))
    finalimg.show()

    fig = visualize_optim(Record, titlestr="cmasteps %d gradsteps %d refinesteps %d Hbasis %s, CMApostAdam %d"%(cmasteps, gradsteps, finalgrad, basis, CMApostAdam))
    fig.savefig(join(savedir, "%straj_H%s%s_%d_dsim_%.3f_L1_%.3f.jpg" % (imgnm, basis, "_postAdam" if CMApostAdam else "",
                                                                        RND, scores_final.min(), L1score_final.min())))
    np.savez(join(savedir, "%soptim_data_%d.npz" % (imgnm, RND)), codes=latent_code.cpu().detach().numpy(),
             Record=Record, dsims=scores_final, L1dsims=L1score_final)

    return imgs.cpu().detach().numpy(), latent_code.cpu().detach().numpy(), scores_final, L1score_final, Record


def visualize_optim(Record, titlestr="", savestr=""):
    """ Pretty visualization function designed for Basin CMA method 
    Show the CMA steps and Adam gradient steps together.

    Input: 
        Record: Last output generated from `BasinCMA` function. 
    """
    fig, ax = plt.subplots()
    L1_cma_tr = np.array(Record["L1cma"])
    dsim_cma_tr = np.array(Record["dsimcma"])
    L1_adam_tr = np.array(Record["L1Adam"])
    dsim_adam_tr = np.array(Record["dsimAdam"])
    dsim_tr = np.array(Record["dsimrefine"])
    L1_tr = np.array(Record["L1refine"])
    nos_norm = np.array(Record["noisenorm"])
    cls_norm = np.array(Record["classnorm"])
    # CMA steps are showed in scatter. x starts from < 0
    cma_steps, popsize = L1_cma_tr.shape
    xticks_arr = np.arange(-cma_steps*10, 0, 10)[:,np.newaxis].repeat(popsize, 1)
    ax.scatter(xticks_arr, L1_cma_tr, color="blue", s=15)
    ax.scatter(xticks_arr, dsim_cma_tr, color="green", s=15)
    ax.scatter(xticks_arr+0.2, L1_adam_tr, color="blue", s=15)
    ax.scatter(xticks_arr+0.2, dsim_adam_tr, color="green", s=15)
    # Adam steps are showed in line plot. x starts from 0
    ax.plot(dsim_tr, label="dsim", color="green", alpha=0.7)
    ax.plot(L1_tr, label="L1", color="blue", alpha=0.7)
    ax.set_ylabel("Image Dissimilarity", color="blue", fontsize=14)
    plt.legend()
    # vector norm is showed in 2nd y plot.
    ax2 = ax.twinx()
    ax2.plot(nos_norm, color="orange", label="noise", alpha=0.7)
    ax2.plot(cls_norm, color="magenta", label="class", alpha=0.7)
    ax2.set_ylabel("L2 Norm", color="red", fontsize=14)
    plt.legend()
    plt.title(titlestr)
    plt.show()
    return fig


def load_BigGAN_basis():
    """load and return different vector basis for the BigGAn space 
    Output: 
        basisdict: a dictionary with following entries. 
            "all": Eigenvectors for the full Hessian of the space.
            "sep": Eigenvector of noise and class concatenated. 
            "none": normal coordinate basis. 
    """
    _, _, evc_clas = load_Haverage("BigGAN", spec="class")
    _, _, evc_nois = load_Haverage("BigGAN", spec="noise")
    _, _, evc_all = load_Haverage("BigGAN", spec="all")
    evc_sep = torch.from_numpy(block_diag(evc_nois, evc_clas)).cuda()
    # evc_clas = torch.from_numpy(evc_clas)#.cuda()
    # evc_nois = torch.from_numpy(evc_nois)#.cuda()
    evc_all = torch.from_numpy(evc_all).cuda()
    evc_none = torch.eye(256).cuda()
    basisdict = {"all": evc_all, "sep": evc_sep, "none": evc_none}
    return basisdict  # evc_all, evc_sep, evc_none

if __name__ is "__main__":
    from .GAN_utils import loadBigGAN, BigGAN_wrapper
    BGAN = loadBigGAN()
    G = BigGAN_wrapper(BGAN)
    ImDist = LPIPS(net="squeeze", ).cuda()
    ImDist.requires_grad_(False)
    basisdict = load_BigGAN_basis()
    imroot = r"src"
    imgnm = r"monkey.jpg"
    saveroot = r"results"
    srcimg = plt.imread(join(imroot, imgnm))
    srcimg_rsz = crop_rsz(srcimg, crop_param="center", )
    srctsr_rsz = ToTensor()(srcimg_rsz).unsqueeze(0).float().cuda()
    monkey_vec = G.BigGAN.embeddings(torch.tensor(one_hot_from_int(373)).cuda())[0].cpu()
    imgs_final, codes_final, scores_final, L1score_final, Record = BasinCMA_BigGAN(srctsr_rsz, G, ImDist,
           cmasteps=80, gradsteps=0, finalgrad=400, batch_size=30, basis="all", basisvec=basisdict["all"],
           CMApostAdam=False, savedir=saveroot, imgnm=imgnm+"_cma_pen", classvec_init=monkey_vec,
           L2penalty=(0.09, 0), classpenalty=0.4)
    # imgs_final, codes_final, scores_final, L1score_final, Record = BasinCMA_BigGAN(srctsr_rsz, G, ImDist,
    #    cmasteps=10, gradsteps=10, finalgrad=500, batch_size=4, basis="all", basisvec=basisdict["all"],
    #    CMApostAdam=False, savedir=saveroot, imgnm=imgnm)
    #%%
    from GAN_utils import loadStyleGAN2, StyleGAN2_wrapper
    imroot = r"src"
    resdir = r"results"
    SGAN = loadStyleGAN2("ffhq-512-avg-tpurun1.pt")
    G = StyleGAN2_wrapper(SGAN)
    G.StyleGAN.requires_grad_(False)
    G.StyleGAN.eval()
    G.use_wspace(False)
    H, eva, evc = load_Haverage("StyleGAN2-Face512_Z", descend=True)
    evctsr = torch.tensor(evc).cuda().float()
    srcimg = plt.imread(join(imroot, "BXW_400x400.jpg"))
    srctsr = ToTensor()(srcimg).unsqueeze(0)
    srctsr_rsz = F.interpolate(srctsr, [256, 256]).cuda()
    init_vect = G.sample_vector(4, )
    refvecs, refimgs, EuclidParam, mse_col, dsim_col = img_project(srctsr_rsz, G, mask_LPIPS, imgnm="BXW",
               stepn=250, sampN=4, initvec=init_vect, wspace=False, 
               hess_precon=False, evctsr=evctsr,
               initEuclid=None, regist_freq=10, euclid_tfm=True, tfm_target=False, 
               paramstr="Z_noH_Eucl", figdir=figdir)
#%%
