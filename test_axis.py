import torch
from PIL import Image
import argparse
from models.e4e.psp import pSp
import os
from core.GAN_utils import StyleGAN2_wrapper, loadStyleGAN2
import numpy as np
from tqdm import tqdm
# from cv2 import line
# from core.geometry_utils import LExpMap
# from numpy.lib.function_base import interp
# from criteria.id_loss import IDLoss
# from math import trunc
# from numpy.lib.npyio import load
# from core import get_full_hessian, hessian_compute, save_imgrid, show_imgrid
# from core.GAN_utils import StyleGAN2_wrapper, loadStyleGAN2

# import lpips
# from matplotlib import pyplot as plt
# from torch.nn import CosineSimilarity
# from torchvision.transforms import Normalize
# #from decalib.deca import Deca
# from matplotlib import pyplot as plt
# # from core.estimate_pose import Estimate_pose
# # from daca_loss import Deca
# #from torch import autograd
# import numpy as np
# from core.hessian_axis_visualize import vis_eigen_action, vis_eigen_explore, vis_distance_curve
# from tqdm import tqdm
# from criteria.id_loss import IDLoss
# from torch.nn.functional import interpolate,cosine_similarity




def get_latents(net, x):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    return codes


def get_test_image():
    pass


if __name__ == '__main__':

    
    ##################################### init ################################
    ckpt = torch.load('./pretrained_models/e4e_ffhq_encode.pt')
    opts = ckpt['opts']
    opts['stylegan_size'] = 1024
    opts['checkpoint_path'] = './pretrained_models/e4e_ffhq_encode.pt'
    opts = argparse.Namespace(**opts)
    net = pSp(opts).cuda()
    stylegan2 = loadStyleGAN2()
    G = StyleGAN2_wrapper(stylegan2)
    G.use_wspace()
    result = np.load('./pretrained_models/pose.npz')
    evc_FI = result['evc_FI']
    eva_FI = result['eva_FI']
    print('Finish init!')


    # result = np.load('./pretrained_models/result_deca_pose_div_final.npy',allow_pickle=True).item()
    # evc_FI = torch.from_numpy(result['eigen_vector']).cuda().T.reshape(-1,18,512)
    # evc_FI = evc_FI.reshape(-1,18*512).T.float()
    # eva_FI = result['eigen_value']
    # evc_FI = gram_schmidt(evc_FI)
    # evc_FI =  evc_FI.data.cpu().numpy()
    # np.savez('./pretrained_models/pose.npz',evc_FI=evc_FI, eva_FI=eva_FI)
    # evc_FI =  ID_Basis.data.cpu().numpy()
    ##########################################################################
   
    with torch.no_grad():
        for i,imgname in tqdm(enumerate(image_name[:1000])):
            #prefix = str(i).zfill(5)
            #img_path = os.path.join(img_root, f'{i}.jpg')
            img_path = os.path.join(celebA, imgname)
            if not os.path.exists(img_path):
                print('continue:')
                continue
            #img = np.array(Image.open(img_path).resize((256, 256))) / 255.
            img = np.array(Image.open(img_path).resize((256, 256))) / 255.
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
            img_tensor = img_tensor.sub_(0.5).div(0.5).float().cuda()
            #print(img_tensor.dtype)
            #exit()
            #latent_editor_wrapper.get_single_ganspace_edits()
            feat = get_latents(net, img_tensor)
            #print(feat.shape)
            # feat = G.mean_latent
            #feat = G.sample_vector()
            # feat = feat.unsqueeze(0).repeat(1, 18, 1)
            #basis_preprocess()
            # result = np.load('pose_orthe.npy',allow_pickle=True)
            # evc_FI = torch.from_numpy(result)

            
            
            
            # feat = result['feat']
            # feat = G.sample_vector()
            feat = feat.detach().cpu().numpy()
            
            # print(evc_FI.shape)
            # exit()
            # interpolate(G, ImDist, feat=torch.from_numpy(feat).cuda())
            # exit()
            #print('aaa',eva_FI.shape)
            #deca = Deca()
            rown = 5
            eiglists = range(2)
            #eiglists = [2]
            
            # for j in eiglists:
                #eiglist = range(10)
                #eiglist = [j]
                # max_dist = 2 # exp
                # max_dist = 10 # pose
            print(evc_FI)
            max_dist = 10
            step_size = max_dist * 2. / (rown - 1.)
            #print('sss',evc_FI.shape)
            mtg, codes_all, imgs = vis_eigen_explore(feat, evc_FI, eva_FI, G, ImDist=None, eiglist=eiglists, transpose=False,
            maxdist=max_dist, scaling=None, rown=rown, sphere=False, distrown=15, RND=i,
                save=True, figdir='result_pose_id2', namestr="demo")
            imgs = torch.Tensor(imgs).permute(0,3,1,2).cuda()
            #print(imgs)
            #print(imgs.shape)
            # exit()
            # a = deca.extract_param(imgs)
            # id2 = arcface.extract_feats(interpolate(imgs,size=(256,256)))
            # id_x = cosine_similarity(id2,id2[rown//2:rown//2+1]).detach().cpu().numpy()
            # angles_x = ((a['pose'][:,:3]*180/np.pi-a['pose'][rown//2,:3]*180/np.pi)).detach().cpu().numpy()
            # # angles_x = ((a['light'][:,:]-a['light'][6,:])**2).mean().detach().cpu().numpy()
            # #angles_x += ((a['exp'][:,3:]-a['exp'][6,3:])**2).mean().detach().cpu().numpy()
            # id_xs.append(id_x)
            # angles_xs.append(angles_x)
            # #print(np.array(angles_xs),np.array(id_xs))
            # # import matplotlib.pyplot as plt
            # # plt.figure()
            # # #print(np.array(angles_xs).shape)
            # #vals = ((np.array(angles_xs)[...,:2]**2).sum(-1)**0.5).reshape(-1)
            
            # # nbins = 100
            # # bins  =  np.linspace(-50, 50, nbins+1)
            # # #bins = np.linspace(0, 1, nbins+1)
            # # id_vals = 1-np.array(id_xs).reshape(-1)
            # ratio =  ((np.array(angles_x)[...,:2]**2).sum(-1)**0.5).reshape(-1)/(1-np.array(id_x).reshape(-1)+1e-12)
            # #print(ratio.mean())
            # #ratios.append(ratio)
            # if ratio.mean()>max_ratio:
            #     max_i = i
            #     max_ratio = ratio.mean()
            # print(max_i,max_ratio)
                
            # ind = np.digitize(vals, bins)
            # result = [func_mean(id_vals[ind == j]) for j in range(1, nbins)]
            # #print(len(result),len(bins))
            # #print(result)
            # #n, _ = np.histogram()
            # plt.plot(bins[0:-2],result)
            # #plt.his
            # plt.xlim(-45,45)
            # plt.ylim(0,1)
            # plt.grid()
            # plt.savefig('2.png')
        #np.save(ratios)
        # np.savez('result_our_basis_id.npz',id_x = id_vals, angles_xs = vals, ratios=np.array(ratios))
        
            #exit(0)
        # np.save('select_exp_multi.npy',np.array(angles_xs))
        # exit()
    # id_xs = np.array(id_xs)
    # angles_xs = np.array(angles_xs)
    # print(id_xs.shape,angles_xs.shape)
    # np.savez('result2.npz',id_x = id_xs, angles_xs = angles_xs)
        #exit()
    



    

    # vis_distance_curve(feat, evc_FI, eva_FI, G, ImDist, eiglist=eiglist,
    #                 maxdist=max_dist, rown=rown, sphere=False, distrown=15, namestr="demo")
