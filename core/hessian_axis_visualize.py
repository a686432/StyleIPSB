"""Visualize the Visual contents of the Hessian Eigenvectors
Achieve functions like:

- Visualize image change along the Hessian eigen vectors for a reference vector. 
    - `vis_eigen_frame`: Useful to plot the full spectrum
    - `vis_eigen_explore`: Useful to plot selected eigen axes
    - `vis_eigen_explore_row`: Save montage figure for each row. 
- Compare effect of travelling along an eigenvector at multiple reference vectors. 
    - `vis_eigen_action`
    - `vis_eigen_action_row`: Save montage figure for each row. 
- Plot the image distance curve, as a function of latent distance from reference image.
    - `vis_distance_curve` 

All of these have similar APIs. 

Documentation added. 
Binxu Wang. 2021 Mar.9th
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from time import time
from os.path import join
from imageio import imwrite, imsave
from PIL import Image
from .torch_utils import show_npimage, show_imgrid
from .build_montages import build_montages, color_framed_montages
from .geometry_utils import SLERP, LERP, LExpMap, SExpMap

def vis_eigen_frame(eigvect_avg, eigv_avg, G, ref_code=None, eiglist=None, eig_rng=(0, 4096), page_B=50, transpose=True,
                    maxdist=120, rown=7, sphere=False, 
                    figdir="", RND=None, namestr="", ):
    """ Visualize image content of moving along Hessian Eigen frame at one `ref_code`
    Go through spectrum in batch, and plot `page_B` number of directions in a page, `rown` in a row.
    m
    Input:
        eigvect_avg: Eigenvectors to visualize, numpy array, shape (N, s)
        eigv_avg: Corresponding Eigenvalues. numpy array, shape (s, )
        G: A wrapped up Generator. 
        ref_code: Reference code z_0, of shape (1, N). Default to be the origin in N d space, np.zeros(1, N). 
    
    Printing parameters: 
        eig_rng: another form to specify range or eigen pairs to visualize. A tuple of 2 integers. 
        eiglist: a list ot eigen indices to visualize, start from 0. 
        page_B: Page limit, number of reference vectors to show on each page / figure. 
        transpose: True means each column shows an eigenvector; False means each row shows an eigenvector.

    Exploration parameters:
        sphere: Bool, True to do spherical exploration, False for linear exploration. 
        maxdist: A scaler, maximal distance to explore in the latent space. if sphere is False, then this is L2 distance along the eigenvector
            if sphere is True, then this is the angle (in rad, [0, pi/2]) to turn towards the eigenvector
        rown: An integer. Number of image to generate for each eigenvector. Normally an odd integer to make the center image the reference image.
            if transpose is True, this is the number of images each row. 
    
    Save parameters:
        figdir: Directory to output figures. 
        namestr: Prefix for the saved figure.
        RND: Random number identifier for saved figure. (same for each sequence. default to be 4 digits RND)

    Output:
        mtg: montage image, np array
        codes_col: list of codes for visualization.
    """
    if ref_code is None: ref_code = np.zeros((1, eigvect_avg.shape[0]))
    if RND is None: RND = np.random.randint(10000)
    if eiglist is None: eiglist = list(range(eig_rng[0], eig_rng[1]))
    t0 = time()
    csr = 0
    codes_page = []
    codes_col = []
    for idx, eigi in enumerate(eiglist):  # range(eig_rng[0]+1, eig_rng[1]+1):
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        codes_page.append(interp_codes) # code to visualize on a page
        if (idx == csr + page_B - 1) or idx + 1 == len(eiglist): # print a page
            codes_all = np.concatenate(tuple(codes_page), axis=0)
            img_page = G.render(codes_all)
            mtg = build_montages(img_page, (256, 256), (rown, idx - csr + 1), transpose=transpose)[0]
            # Image.fromarray(np.uint8(mtg * 255.0)).show()
            # imsave(join(figdir, "%d-%d.jpg" % (csr, eigi)), np.uint8(mtg * 255.0))
            imsave(join(figdir, "%s_%d-%d_%.e~%.e_%04d.jpg" %
            (namestr, eiglist[csr]+1, eigi+1, eigv_avg[-eiglist[csr]-1], eigv_avg[-eigi], RND)), np.uint8(mtg * 255.0))
            codes_col.append(codes_all)
            codes_page = []
            print("Finish printing page eigen %d-%d (%.1fs)"%(eiglist[csr], eigi, time()-t0))
            csr = idx + 1
    return mtg, codes_col

def vis_eigen_explore(ref_code, eigvect_avg, eigv_avg, G, ImDist=None, eiglist=[1,2,4,7,16], transpose=True, 
                      maxdist=120, scaling=None, rown=5, sphere=False, distrown=19, 
                      save=True, figdir="", RND=None, namestr=""):
    """ Visualize image content of moving along eigenvectors at a `ref_code`
    Plot image distance curve for these directions if ImDist provided.
    
    Input:
        ref_code: Reference code z_0, of shape (1, N) or (N, ). Default to be the origin in N d space, np.zeros(1, N). 
        eigvect_avg: Eigenvectors to visualize, numpy array, shape (N, s)
        eigv_avg: Corresponding Eigenvalues. numpy array, shape (s, )
        G: A wrapped up Generator. 
        ImDist: Image distance function. If specified then compute the distance curve as well. 
            distrown: parameter for ploting image distance curve.
    
    Printing parameters: 
        eig_rng: another form to specify range or eigen pairs to visualize. A tuple of 2 integers. 
        eiglist: a list ot eigen indices to visualize, start from 0. 
        page_B: Page limit, number of reference vectors to show on each page / figure. 
        transpose: True means each column shows an eigenvector; False means each row shows an eigenvector.

    Exploration parameters:
        sphere: Bool, True to do spherical exploration, False for linear exploration. 
        maxdist: A scaler, maximal distance to explore in the latent space. if sphere is False, then this is L2 distance along the eigenvector
            if sphere is True, then this is the angle (in rad, [0, pi/2]) to turn towards the eigenvector
        rown: An integer. Number of image to generate for each eigenvector. Normally an odd integer to make the center image the reference image.
            if transpose is True, this is the number of images each row. 
        **scaling**: An array specifying the scaling factor for each eigvect to plot, s.t. actual maximal travelling distance or angle will be 
            (-scaling[i] * maxdist, scaling[i] * maxdist).  Array or list of shape (len(eiglist), )  
    
    Save parameters:
        figdir: Directory to output figures. 
        namestr: Prefix for the saved figure.
        RND: Random number identifier for saved figure. (same for each sequence. default to be 4 digits RND)

    Output:
        mtg: montage image, np array
        codes_col: list of codes for visualization.
        distmat, fig: will be returned if the image distance plot is done. 
    """
    if RND is None: RND = np.random.randint(50000)
    if eiglist is None: eiglist = list(range(len(eigv_avg)))
    if scaling is None: scaling = np.ones(len(eigv_avg))
    t0 = time()
    codes_page = []
    for idx, eigi in enumerate(eiglist):  
        scaler = scaling[idx]
        #print('sss',eigvect_avg[:, -eigi-1].shape)
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist*scaler, maxdist*scaler))
        else:
            interp_codes = SExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist*scaler, maxdist*scaler))
        #print(ref_code.shape,interp_codes.shape)
        interp_codes = interp_codes.reshape(interp_codes.shape[0],18,512)
        #print(ref_code,eigvect_avg)
        # print(ref_code.shape,interp_codes.shape)
        # c = interp_codes[:,np.newaxis,:].repeat(18,axis=1)
        # interp_codes[:,:8,:] = ref_code[0][:8,:]
        # interp_codes[...,:4,:] = ref_code[0][:4,:]
        #interp_codes[...,3:,:] = ref_code[0][3:,:]
        
        codes_page.append(interp_codes)
    codes_all = np.concatenate(tuple(codes_page), axis=0)
    #print(codes_all)
    # exit()
    #print(codes_all.shape)
    img_page = G.render(codes_all)
    # print(img_page[0].shape)
    # exit()
    mtg = build_montages(img_page, (256, 256), (rown, len(eiglist)), transpose=transpose)[0]
    if save:
        print('aaaa')
        imsave(join(figdir, "%s_%d-%d_%04d.jpg" % (namestr, eiglist[0]+1, eiglist[-1]+1, RND)), np.uint8(mtg * 255.0))
        #plt.imsave(join(figdir, "%s_%d-%d_%04d.pdf" % (namestr, eiglist[0]+1, eiglist[-1]+1, RND)), mtg, )
    # else:
    #     show_npimage(mtg)

    print("Finish printing page (%.1fs)" % (time() - t0))
    if ImDist is not None: # if distance metric available then compute this
        distmat, ticks, fig = vis_distance_curve(ref_code, eigvect_avg, eigv_avg, G, ImDist, eiglist=eiglist,
	        maxdist=maxdist, rown=rown, distrown=distrown, sphere=sphere, figdir=figdir, RND=RND, namestr=namestr, )
        return mtg, codes_all, distmat, fig
    else:
        return mtg, codes_all,img_page

def vis_eigen_explore_row(ref_code, eigvect_avg, eigv_avg, G, eiglist=[1,2,4,7,16], transpose=True, 
     maxdist=120, rown=5, sphere=False, 
     save=True, figdir="", RND=None, namestr="", indivimg=False):  
    """ Same as `vis_eigen_explore` but save each each row separately. 
    Additional Parameter:
        indivimg: Save individual image separately. 
    """
    if RND is None: RND = np.random.randint(10000)
    if eiglist is None: eiglist = list(range(len(eigv_avg)))
    t0 = time()
    codes_page = []
    mtg_col = []
    ticks = np.linspace(-maxdist, maxdist, rown)
    for idx, eigi in enumerate(eiglist):  
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvect_avg[:, -eigi-1], rown, (-maxdist, maxdist))
        codes_page.append(interp_codes)
        img_page = G.render(interp_codes)
        mtg = build_montages(img_page, (256, 256), (rown, 1), transpose=transpose)[0]
        if save:
            imsave(join(figdir, "%s_eig%d_%04d.jpg" % (namestr, eigi+1, RND)), np.uint8(mtg * 255.0))
            plt.imsave(join(figdir, "%s_eig%d_%04d.pdf" % (namestr, eigi+1, RND)), mtg, )
        else:
            show_npimage(mtg)
        mtg_col.append(mtg)
        if indivimg and save: # save individual images. 
            for deviation, img in zip(ticks, img_page):
                imsave(join(figdir, "%s_eig%d_%.1e_%04d.jpg" % (namestr,eigi+1, deviation, RND)), np.uint8(img * 255.0))
    codes_all = np.concatenate(tuple(codes_page), axis=0)
    print("Finish printing page (%.1fs)" % (time() - t0))
    return mtg_col, codes_all

def vis_distance_curve(ref_code, eigvect_avg, eigvals_avg, G, ImDist, eiglist=[1,2,4,7,16],
	    maxdist=0.3, rown=3, distrown=19, sphere=False, 
        figdir="", RND=None, namestr="", ):
    """Compute image distance to reference as a function of explore distance. 
    Input:
        ref_code: Reference code z_0, of shape (1, N). Default to be the origin in N d space, np.zeros(1, N). 
        eigvect_avg: Eigenvectors to visualize, numpy array, shape (N, s)
        eigv_avg: Corresponding Eigenvalues. numpy array, shape (s, )
        G: A wrapped up Generator. 
        ImDist: image distance function, like LPIPS
    
    Printing parameters: 
        eiglist: a list ot eigen indices to visualize,  start from 0. 
        distrown: number of samples to compute distance curve for each eigenvector
        rown: number of x ticks to show for the curve. 

    Exploration parameters:
        sphere: Bool, True to do spherical exploration, False for linear exploration. 
        maxdist: A scaler, maximal distance to explore in the latent space. if sphere is False, then this is L2 distance along the eigenvector
            if sphere is True, then this is the angle (in rad, [0, pi/2]) to turn towards the eigenvector
        
    Save parameters:
        figdir: Directory to output figures. 
        namestr: Prefix for the saved figure.
        RND: Random number identifier for saved figure. (same for each sequence. default to be 4 digits RND)

    Output:
        distmat: distance curve for each eigenvector, concatenated in a matrix. 
        ticks: x ticks for the distance matrix. 
        fig: Figure handle
    """
    if RND is None: RND = np.random.randint(10000)
    refimg = G.visualize_batch_np(ref_code.reshape(1, -1))
    codes_page = []
    ticks = np.linspace(-maxdist, maxdist, distrown, endpoint=True)
    visticks = np.linspace(-maxdist, maxdist, rown, endpoint=True) # actual x ticks to show.
    for idx, eigi in enumerate(eiglist):  
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvect_avg[:, -eigi - 1], distrown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvect_avg[:, -eigi - 1], distrown, (-maxdist, maxdist))
        codes_page.append(interp_codes)
        # if (idx == csr + page_B - 1) or idx + 1 == len(eiglist):
    codes_all = np.concatenate(tuple(codes_page), axis=0)
    img_page = G.visualize_batch_np(codes_all)
    with torch.no_grad():
        dist_all = ImDist(refimg.cuda(), img_page.cuda()).squeeze().cpu()
    # print(dist_all)
    distmat = dist_all.reshape(-1, distrown).numpy()
    fig = plt.figure(figsize=[5, 3])
    for idx, eigi in enumerate(eiglist):
        plt.plot(ticks, distmat[idx, :], label="eig%d %.E" % (eigi + 1, eigvals_avg[-eigi - 1]), lw=2.5, alpha=0.7)
    plt.xticks(visticks)
    plt.ylabel("Image distance")
    plt.xlabel("L2 in latent space" if not sphere else "Angle (rad) in latent space")
    plt.legend()
    plt.subplots_adjust(left=0.14, bottom=0.14)
    plt.savefig(join(figdir, "%s_imdistcrv_%04d.jpg" % (namestr, RND)) )
    plt.savefig(join(figdir, "%s_imdistcrv_%04d.pdf" % (namestr, RND)) )
    plt.show()
    return distmat, ticks, fig

def vis_eigen_action(eigvec, ref_codes, G, maxdist=120, rown=7, sphere=False, 
                    page_B=50, transpose=True, save=True, figdir="", namestr="", RND=None):
    """ Visualize action of an eigenvector on multiple reference images.
    
    Input:
        ref_codes: Reference codes z_0, of shape (r, N) or (N, r). r is the number of z_0 to compare.
            Default to be the origin in N d space, np.zeros(1, N). 
        eigvect_avg: Eigenvector to visualize, assume there is only one. numpy array, shape (N, 1) or (N,) or (1,N)
        G: A wrapped up Generator. 
    
    Printing parameters: 
        page_B: Page limit, number of reference vectors to show on each page / figure. 
        transpose: True means each column shows an eigenvector; False means each row shows an eigenvector.

    Exploration parameters:
        sphere: Bool, True to do spherical exploration, False for linear exploration. 
        maxdist: A scaler, maximal distance to explore in the latent space. if sphere is False, then this is L2 distance along the eigenvector
            if sphere is True, then this is the angle (in rad, [0, pi/2]) to turn towards the eigenvector
        rown: An integer. Number of image to generate for each eigenvector. Normally an odd integer to make the center image the reference image.
            if transpose is True, this is the number of images each row. 
        
    Save parameters:
        save: Bool. save figure or just show.
        figdir: Directory to output figures. 
        namestr: Prefix for the saved figure.
        RND: Random number identifier for saved figure. (same for each sequence. default to be 4 digits RND)

    Output:
        mtg: montage image, np array
        codes_col: list of codes for visualization.
    """
    if ref_codes is None:
        ref_codes = np.zeros(eigvec.size)
    if RND is None: RND = np.random.randint(10000)
    reflist = list(ref_codes)
    t0 = time()
    csr = 0
    codes_page = []
    codes_col = []
    for idx, ref_code in enumerate(reflist):  
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        codes_page.append(interp_codes)
        if (idx == csr + page_B - 1) or idx + 1 == len(reflist):
            codes_all = np.concatenate(tuple(codes_page), axis=0)
            img_page = G.render(codes_all)
            mtg = build_montages(img_page, (256, 256), (rown, idx - csr + 1), transpose=transpose)[0]
            if save:
                imsave(join(figdir, "%s_ref_%d-%d_%04d.jpg" %
                        (namestr, csr, idx, RND)), np.uint8(mtg * 255.0))
            else:
                show_npimage(mtg)
            codes_col.append(codes_all)
            codes_page = []
            print("Finish printing page vector %d-%d (%.1fs)"%(csr, idx, time()-t0))
            csr = idx + 1
    return mtg, codes_col

def vis_eigen_action_row(eigvec, ref_codes, G, maxdist=120, rown=7, sphere=False, 
                    transpose=True, save=True, figdir="", namestr="", RND=None, indivimg=False,):
    """Same as `vis_eigen_action` just print each row separately
    Additional Parameter:
        indivimg: Save individual image separately. 
    """
    if ref_codes is None:
        ref_codes = np.zeros(eigvec.size)
    if RND is None: RND = np.random.randint(10000)
    reflist = list(ref_codes)
    t0 = time()
    codes_col = []
    mtg_col = []
    ticks = np.linspace(-maxdist, maxdist, rown)
    for idx, ref_code in enumerate(reflist):  
        if not sphere:
            interp_codes = LExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        else:
            interp_codes = SExpMap(ref_code, eigvec, rown, (-maxdist, maxdist))
        img_page = G.render(interp_codes)
        mtg = build_montages(img_page, (256, 256), (rown, 1), transpose=transpose)[0]
        if save:
            imsave(join(figdir, "%s_ref_%d_%04d.jpg" %
                    (namestr, idx, RND)), np.uint8(mtg * 255.0))
        else:
            show_npimage(mtg)
        codes_col.append(interp_codes)
        mtg_col.append(mtg)
        if indivimg:
            for div, img in zip(ticks, img_page):
                imsave(join(figdir, "%s_ref_%d_%.1e_%04d.jpg" % (namestr, idx, div, RND)), np.uint8(img * 255.0))
        print("Finish printing along vector %d (%.1fs)"%(idx, time()-t0))
    return mtg_col, codes_col

if __name__ == "__main__":
    from .GAN_utils import BigGAN_wrapper, loadBigGAN, upconvGAN
    from lpips import LPIPS
    ImDist = LPIPS(net="squeeze")
    #%% FC6 GAN on ImageNet
    G = upconvGAN("fc6")
    G.requires_grad_(False).cuda()  # this notation is incorrect in older pytorch
    #%% Average Hessian for the Pasupathy Patches
    out_dir = r"E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace"
    with np.load(join(out_dir, "Pasu_Space_Avg_Hess.npz")) as data:
        # H_avg = data["H_avg"]
        eigvect_avg = data["eigvect_avg"]
        eigv_avg = data["eigv_avg"]
    figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec"
    vis_eigen_frame(eigvect_avg, eigv_avg, figdir=figdir)
    #%% Average hessian for the evolved images
    out_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
    with np.load(join(out_dir, "Evolution_Avg_Hess.npz")) as data:
        # H_avg = data["H_avg"]
        eigvect_avg = data["eigvect_avg"]
        eigv_avg = data["eigv_avg"]
    figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec_Evol"
    vis_eigen_frame(eigvect_avg, eigv_avg, figdir=figdir)
    #%% use the initial gen as reference code, do the same thing
    out_dir = r"E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace"
    with np.load(join(out_dir, "Texture_Avg_Hess.npz")) as data:
        # H_avg = data["H_avg"]
        eigvect_avg = data["eigvect_avg"]
        eigv_avg = data["eigval_avg"]
    #%%
    code_path = r"D:\Generator_DB_Windows\init_population\texture_init_code.npz"
    with np.load(code_path) as data:
        codes_all = data["codes"]
    ref_code = codes_all.mean(axis=0, keepdims=True)
    #%%
    figdir = r"E:\OneDrive - Washington University in St. Louis\HessTune\HessEigVec_Text"
    vis_eigen_frame(eigvect_avg, eigv_avg, G, figdir=figdir, ref_code=ref_code,
                    maxdist=120, rown=7, eig_rng=(0, 4096))
    #%%
    figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN"
    vis_eigen_frame(eigvect_avg, eigv_avg, ref_code=None, figdir=figdir, page_B=50,
                    eiglist=[0,1,2,5,10,20,30,50,100,200,300,400,600,800,1000,2000,3000,4000], maxdist=240, rown=5,
                    transpose=False)
    #%%
    vis_eigen_action(eigvect_avg[:, -5], np.random.randn(10,4096), G, figdir=figdir, page_B=50,
                        maxdist=20, rown=5, transpose=False)
    #%%
    vis_eigen_action(eigvect_avg[:, -5], None, G, figdir=figdir, page_B=50,
                        maxdist=20, rown=5, transpose=False)
    #%% BigGAN on ImageNet Class Specific

    from torchvision.transforms import ToPILImage
    BGAN = loadBigGAN("biggan-deep-256").cuda()
    BG = BigGAN_wrapper(BGAN)
    EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
    #%%
    figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
    Hessdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigGAN"
    data = np.load(join(Hessdir, "H_avg_1000cls.npz"))
    eva_BG = data['eigvals_avg']
    evc_BG = data['eigvects_avg']
    evc_nois = data['eigvects_nois_avg']
    evc_clas = data['eigvects_clas_avg']
    #%%
    imgs = BG.render(np.random.randn(1, 256)*0.06)
    #%%
    eigi = 5
    refvecs = np.vstack((EmbedMat[:,np.random.randint(0, 1000, 10)], 0.5*np.random.randn(128,10))).T
    vis_eigen_action(evc_BG[:, -eigi], refvecs, figdir=figdir, page_B=50, G=BG,
                     maxdist=0.5, rown=5, transpose=False, namestr="eig%d"%eigi)
    #%% Effect of eigen vectors within the noise space
    eigi = 3
    tanvec = np.hstack((evc_nois[:, -eigi], np.zeros(128)))
    refvecs = np.vstack((EmbedMat[:,np.random.randint(0, 1000, 10)], 0.5*np.random.randn(128,10))).T
    vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                     maxdist=2, rown=5, transpose=False, namestr="eig_nois%d"%eigi)
    #%%
    eigi = 3
    tanvec = np.hstack((np.zeros(128), evc_clas[:, -eigi]))
    refvecs = np.vstack((EmbedMat[:,np.random.randint(0, 1000, 10)], 0.5*np.random.randn(128,10))).T
    vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                     maxdist=0.4, rown=5, transpose=False, namestr="eig_clas%d"%eigi)
    #%%
    eigi = 120
    tanvec = np.hstack((np.zeros(128), evc_clas[:, -eigi]))
    refvecs = np.vstack((EmbedMat[:, np.random.randint(0, 1000, 10)], 0.5*np.random.randn(128,10))).T
    vis_eigen_action(tanvec, refvecs, figdir=figdir, page_B=50, G=BG,
                     maxdist=2, rown=5, transpose=False, namestr="eig_clas%d"%eigi)

    #%% BigBiGAN on ImageNet
    from .GAN_utils import BigBiGAN_wrapper, loadBigBiGAN
    from torchvision.transforms import ToPILImage
    BBGAN = loadBigBiGAN().cuda()
    BBG = BigBiGAN_wrapper(BBGAN)
    # EmbedMat = BG.BigGAN.embeddings.weight.cpu().numpy()
    #%%
    from .GAN_hessian_compute import hessian_compute, get_full_hessian
    from .hessian_analysis_tools import scan_hess_npz, compute_hess_corr, plot_spectra
    npzdir = r"E:\OneDrive - Washington University in St. Louis\HessGANCmp\BigBiGAN"
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_norm9_(\d*).npz", evakey='eigvals', evckey='eigvects', featkey="vect")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    eigid = 20
    figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\BigBiGAN"
    mtg = vis_eigen_action(eigvec=eigvec_col[12][:, -eigid-1], ref_codes=feat_arr[[12, 0, 2, 4, 6, 8, 10, 12, ], :], G=BBG, maxdist=2, rown=5, transpose=False, namestr="BigBiGAN_norm9_eig%d"%eigid, figdir=figdir)
    #%% StyleGAN2
    from .GAN_hessian_compute import hessian_compute
    from .GAN_utils import loadStyleGAN, StyleGAN_wrapper
    figdir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\StyleGAN2"
    #%% Cats
    modelname = "stylegan2-cat-config-f"
    npzdir = r"E:\Cluster_Backup\StyleGAN2\stylegan2-cat-config-f"
    SGAN = loadStyleGAN(modelname+".pt", size=256, channel_multiplier=2)  #
    G = StyleGAN_wrapper(SGAN)
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP',
                                                           evckey='evc_BP', featkey="feat")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    eigid = 5
    mtg = vis_eigen_action(eigvec=eigvec_col[0][:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, ], :],
                           G=G, maxdist=3, rown=5, transpose=False, namestr="SG2_Cat_eig%d"%eigid, figdir=figdir)
    #%% Animation
    modelname = "2020-01-11-skylion-stylegan2-animeportraits"
    npzdir = r"E:\Cluster_Backup\StyleGAN2\2020-01-11-skylion-stylegan2-animeportraits"
    SGAN = loadStyleGAN(modelname+".pt", size=512, channel_multiplier=2)
    G = StyleGAN_wrapper(SGAN)
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP',
                                                           evckey='evc_BP', featkey="feat")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    eigid = 3
    mtg = vis_eigen_action(eigvec=eigvec_col[0][:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, ], :],
                           G=G, maxdist=10, rown=5, transpose=False, namestr="SG2_anime_eig%d"%eigid, figdir=figdir)
    #%% Faces 256
    modelname = 'ffhq-256-config-e-003810'
    npzdir = r"E:\Cluster_Backup\StyleGAN2\ffhq-256-config-e-003810"
    SGAN = loadStyleGAN(modelname+".pt", size=256, channel_multiplier=1)  #
    G = StyleGAN_wrapper(SGAN)
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP',
                                                           evckey='evc_BP', featkey="feat")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    eigid = 14
    mtg = vis_eigen_action(eigvec=eigvec_col[0][:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, ], :],
                           G=G, maxdist=10, rown=5, transpose=False, namestr="SG2_Face256_eig%d"%eigid, figdir=figdir)
    #%% Faces
    modelname = 'ffhq-256-config-e-003810'
    npzdir = r"E:\Cluster_Backup\StyleGAN2\ffhq-256-config-e-003810"
    SGAN = loadStyleGAN(modelname+".pt", size=256, channel_multiplier=1)  #
    G = StyleGAN_wrapper(SGAN)
    eigval_col, eigvec_col, feat_col, meta = scan_hess_npz(npzdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP',
                                                           evckey='evc_BP', featkey="feat")
    feat_arr = np.array(feat_col).squeeze()
    #%%
    def average_H(eigval_col, eigvec_col):
        """Compute the average Hessian over a bunch of positions"""
        nH = len(eigvec_col)
        dimen = eigval_col.shape[1]
        H_avg = np.zeros((dimen, dimen))
        for iH in range(nH):
            H = (eigvec_col[iH] * eigval_col[iH][np.newaxis, :]) @ eigvec_col[iH].T
            H_avg += H
        H_avg /= nH
        eva_avg, evc_avg = np.linalg.eigh(H_avg)
        return H_avg, eva_avg, evc_avg

    H_avg, eva_avg, evc_avg = average_H(eigval_col, eigvec_col)
    #%%
    maxang = 1.5
    figdir = "E:\Cluster_Backup\StyleGAN2_axis\Face256"
    for eigid in list(range(20))+list(range(20,60,2))+list(range(60,200,4)):
        mtg = vis_eigen_action(eigvec=evc_avg[:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 40, 60], :],
                       G=G, sphere=True, maxdist=1.5, rown=5, transpose=False, namestr="SG2_Face256_AVGeig%d_Sph%.1f"%(eigid, maxang), figdir=figdir)
        print("Finish printing eigenvalue %d"%eigid)
        # if eigid==5:
        #     break

    maxdis = 6
    figdir = "E:\Cluster_Backup\StyleGAN2_axis\Face256"
    for eigid in list(range(20))+list(range(20,60,2))+list(range(60,200,4)):
        mtg = vis_eigen_action(eigvec=evc_avg[:, -eigid-1], ref_codes=feat_arr[[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 40, 60], :],
                               G=G, maxdist=maxdis, rown=7, transpose=False, namestr="SG2_Face256_AVGeig%d_Lin%1.f"%(eigid, maxdis), figdir=figdir)
        print("Finish printing eigenvalue %d"%eigid)

    #%

