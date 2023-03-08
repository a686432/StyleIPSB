# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from .utils.renderer import SRenderY
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .datasets import datasets
from .utils.config import cfg
torch.backends.cudnn.benchmark = True

class DECA(object):
    def __init__(self, config=None, device='cuda'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

    def _setup_renderer(self, model_cfg):
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape+model_cfg.n_tex+model_cfg.n_exp+model_cfg.n_pose+model_cfg.n_cam+model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3 # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i:model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device) 
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail+self.n_cond, out_channels=1, out_scale=model_cfg.max_z, sample_mode = 'bilinear').to(self.device)
        # resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        else:
            print(f'please check model path: {model_path}')
            exit()
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
    
        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        return uv_detail_normals

    def displacement2vertex(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail vertices
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
    
        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        # uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        # uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        detail_faces =  self.render.dense_faces
        return dense_vertices, detail_faces

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:,:,2:] < 0.1).float()
        return vis68

    #@torch.no_grad()
    def encode(self, images):
        batch_size = images.shape[0]
        parameters = self.E_flame(images)
        detailcode = self.E_detail(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['detail'] = detailcode
        codedict['images'] = images
        return codedict

    #@torch.no_grad()
    def decode(self, codedict):
        images = codedict['images']
        batch_size = images.shape[0]
        
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        uv_z = self.D_detail(torch.cat([codedict['pose'][:,3:], codedict['exp'], codedict['detail']], dim=1))
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device) 
        ## projection
        #codedict['cam'] = torch.Tensor([[8., 0., 0.]]).cuda()
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        
        ## rendering
        ops = self.render(verts, trans_verts, albedo, codedict['light'])
        uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
        uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
        uv_texture = albedo*uv_shading

        landmarks3d_vis = self.visofp(ops['transformed_normals'])
        landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)

        ## render shape
        shape_images = self.render.render_shape(verts, trans_verts)
        detail_normal_images = F.grid_sample(uv_detail_normals, ops['grid'], align_corners=False)*ops['alpha_images']
        shape_detail_images = self.render.render_shape(verts, trans_verts, detail_normal_images=detail_normal_images)
        
        ## extract texture
        ## TODO: current resolution 256x256, support higher resolution, and add visibility
        uv_pverts = self.render.world2uv(trans_verts)
        uv_gt = F.grid_sample(images, uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear')
        if self.cfg.model.use_tex:
            ## TODO: poisson blending should give better-looking results
            uv_texture_gt = uv_gt[:,:3,:,:]*self.uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-self.uv_face_eye_mask)*0.7)
        else:
            uv_texture_gt = uv_gt[:,:3,:,:]*self.uv_face_eye_mask + (torch.ones_like(uv_gt[:,:3,:,:])*(1-self.uv_face_eye_mask)*0.7)
            
        ## output
        opdict = {
            'vertices': verts,
            'normals': ops['normals'],
            'transformed_vertices': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'uv_detail_normals': uv_detail_normals,
            'uv_texture_gt': uv_texture_gt,
            'displacement_map': uv_z+self.fixed_uv_dis[None,None,:,:],
        }
        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo
            opdict['uv_texture'] = uv_texture

        visdict = {
            'inputs': images, 
            'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d, isScale=False),
            'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d, isScale=False),
            'shape_images': shape_images,
            'shape_detail_images': shape_detail_images
        }
        if self.cfg.model.use_tex:
            visdict['rendered_images'] = ops['images']
        return opdict, visdict

    def visualize(self, visdict, size=None):
        grids = {}
        if size is None:
            size = self.image_size
        for key in visdict:
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [size, size])).detach().cpu()
        grid = torch.cat(list(grids.values()), 2)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        return grid_image
    
    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['vertices'][i].detach().cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
        util.write_obj(filename, vertices, faces, 
                        texture=texture, 
                        uvcoords=uvcoords, 
                        uvfaces=uvfaces, 
                        normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture = texture[:,:,[2,1,0]]
        normals = opdict['normals'][i].detach().cpu().numpy()
        displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture, self.dense_template)
        util.write_obj(filename.replace('.obj', '_detail.obj'), 
                        dense_vertices, 
                        dense_faces,
                        colors = np.ones_like(dense_colors) * 0.5,
                        inverse_face_order=True)

from decalib.utils.config import cfg as deca_cfg
# from criteria.id_loss import IDLoss
from decalib.datasets.detectors import FAN
# from decalib.deca import DECA
from skimage.transform import estimate_transform 
import numpy as np 
import torch
from torch.nn.functional import grid_sample, affine_grid, interpolate
import face_alignment


class Deca(object):
    def __init__(self):
        self.face_detector = FAN()
        # FLAME encoder
        deca_cfg.model.use_tex = True
        self.deca = DECA(config=deca_cfg)
        self.deca_resolution_inp = 224
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


        # left = bbox[0]; right=bbox[2]
        # top = bbox[1]; bottom=bbox[3]
        # old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
        size = 198
        center = np.array([127,163]) 
        resolution_inp = 256
        #print(center,size)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
        src_pts = 2 * src_pts / 256 - 1
        DST_PTS = 2 * DST_PTS / resolution_inp - 1
        #print(resolution_inp)

        tform = estimate_transform('similarity', src_pts, DST_PTS)
        self.theta = torch.from_numpy(tform._inv_matrix[:2, :]).unsqueeze(0).detach().cuda().float()
    def bbox2point(self,left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def deca_align(self,face_detector, img_tensor, resolution_inp, scale=1.25):
        '''
        align a given img_tensor for DECA
        input:
            img_tensor: shape [1, C, H, W], range [-1, 1]
            resolution_inp: resolution of the output image tensor
        output:
            aligned_img_tensor: shape [1, C, resolution_inp, resolution_inp], range [0, 1]
        '''

        
        
        # img_tensor = img_tensor.requires_grad
        # img_tensor = torch.Tensor(img_tensor.detach().cpu().clone()).cuda()
        # img_tensor.requires_grad = True
        # print(img_tensor.shape)
        
        img_list = []
        flag = True


        for i in range(img_tensor.shape[0]):
            #print(img_tensor.shape)
            #img_tensor = torch.nn.functional.interpolate(img_tensor, size=256)
            img = (img_tensor.detach()[i].permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255
            # print(img.shape)
            # print(face_detector.run(img))
            bbox_type = 'kpt68'
            bbox = face_detector.run(img)[0]
            if isinstance (bbox,int) or  len(bbox) < 4 :
                #img_inp = torch.nn.functional.interpolate(img_tensor[i:i+1], size=resolution_inp)
                size, center = 195.5, [127, 163]
                print("No face detected!")
                flag = False
            else:

                left = bbox[0]; right=bbox[2]
                top = bbox[1]; bottom=bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
                size = int(old_size * scale)
                #print(center,size)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
            src_pts = 2 * src_pts / img.shape[0] - 1
            DST_PTS = 2 * DST_PTS / resolution_inp - 1
            #print(resolution_inp)

            tform = estimate_transform('similarity', src_pts, DST_PTS)
            theta = torch.from_numpy(tform._inv_matrix[:2, :]).unsqueeze(0).detach().cuda().float()
            #print(theta.shape)
            #grid =torch.nn.functional.affine_grid(theta, torch.Size(img_tensor.shape[0],img_tensor.shape[1],resolution_inp,resolution_inp), align_corners=False)
            #exit(0)
            grid = affine_grid(theta, torch.Size((1,img_tensor.shape[1], resolution_inp, resolution_inp)), align_corners=False)
            img_inp = self.grid_sample(img_tensor[i:i+1], grid)
            img_list.append(img_inp)

        img_inp = torch.cat(img_list,dim=0)
        #print(img_inp.min())
        align_info = {}
        align_info['center'] = center
        align_info['size'] = size
        align_info['flag'] = flag 
        # loss = img_inp.mean()
        # loss.backward()
        # print(loss)
        # exit()
        '''
        
        # print(img_tensor.shape)
        align_info = {}
        size, center = 195.5, [127, 163]
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
        src_pts = 2 * src_pts / 256 - 1
        DST_PTS = 2 * DST_PTS / resolution_inp - 1
        # tform = estimate_transform('similarity', src_pts, DST_PTS)
        # theta = torch.from_numpy(tform._inv_matrix[:2, :]).unsqueeze(0).detach().cuda().float()
        
        grid = affine_grid(self.theta.expand(img_tensor.shape[0],self.theta.shape[1],self.theta.shape[2]), torch.Size((img_tensor.shape[0],img_tensor.shape[1], resolution_inp, resolution_inp)), align_corners=False)
        img_inp = self.grid_sample(img_tensor, grid)
        return 0.5 * (img_inp + 1)  ,align_info
        '''
        return 0.5 * (img_inp + 1) ,align_info
    



    def deca_exp_loss(self,img1,img2):
        # img1 = interpolate(img1, (224, 224))
        # img2 = interpolate(img2, (224, 224))
       
        deca_img1,flag1 = self.deca_align(self.face_detector, img1, self.deca_resolution_inp)
        flm_param1 = self.deca.encode(deca_img1)
        deca_img2,flag2 = self.deca_align(self.face_detector, img2, self.deca_resolution_inp)
        flm_param2 = self.deca.encode(deca_img2)
        #print(flm_param1['exp'][0].shape)
        #exit()
        # loss = deca_img1.mean()
        # loss.backward()
        # print(loss)
        # exit()
        print(flm_param1['exp'].shape)
        return ((flm_param1['exp'] - flm_param2['exp'])**2).sum(dim=1) + ((flm_param1['pose'][:,3:] - flm_param2['pose'][:,:3])**2).sum(dim=1)
    def deca_illum_loss(self,img1,img2):
        # img1 = interpolate(img1, (224, 224))
        # img2 = interpolate(img2, (224, 224))
       
        deca_img1,_ = self.deca_align(self.face_detector, img1, self.deca_resolution_inp)
        flm_param1 = self.deca.encode(deca_img1)
        deca_img2,_ = self.deca_align(self.face_detector, img2, self.deca_resolution_inp)
        flm_param2 = self.deca.encode(deca_img2)
        #print(flm_param1['exp'][0].shape)
        #exit()
        # loss = deca_img1.mean()
        # loss.backward()
        # print(loss)
        # # exit()
        # print(flm_param1['light'].shape)
        # exit()
        return ((flm_param1['light'].reshape(-1,27) - flm_param2['light'].reshape(-1,27))**2).mean(dim=1)

    def deca_pose_loss(self,img1,img2):
        # img1 = interpolate(img1, (224, 224))
        # img2 = interpolate(img2, (224, 224))
       
        deca_img1,_ = self.deca_align(self.face_detector, img1, self.deca_resolution_inp)
        flm_param1 = self.deca.encode(deca_img1)
        deca_img2,_ = self.deca_align(self.face_detector, img2, self.deca_resolution_inp)
        flm_param2 = self.deca.encode(deca_img2)
        #print(flm_param1['exp'][0].shape)
        #exit()
        # loss = deca_img1.mean()
        # loss.backward()
        # print(loss)
        # # exit()
        # print(flm_param1['pose'].shape)
        # exit()
        # print(flm_param1['pose'])
        return ((flm_param1['pose'][:,0:2] - flm_param2['pose'][:,0:2])**2).mean(dim=1)


    def deca_loss(self,img1,img2,flag):
        deca_img1,flag1 = self.deca_align(self.face_detector, img1, self.deca_resolution_inp)
        flm_param1 = self.deca.encode(deca_img1)
        flm_param1['pose_flag'] = flag
        #print(flag)
        
        # print(flm_param1['exp'])
        
        result1 = self.deca.decode(flm_param1)
        # flm_param1['pose'][:,:3]=0
        # result11 = self.deca.decode(flm_param1)
        img_v = (deca_img1.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
        #preds = self.fa.get_landmarks(img_v)
       # print(((result1['landmarks2d'][0][17:] - torch.Tensor(preds[0][17:]).to(deca_img1.device))**2).mean())
        deca_img2,flag2 = self.deca_align(self.face_detector, img2, self.deca_resolution_inp)
        flm_param2 = self.deca.encode(deca_img2)
        # para_loss = 1000*((flm_param1['exp'] - flm_param2['exp'])**2).mean() + 1000*((flm_param1['pose'][:,3:] - flm_param2['pose'][:,3:])**2).mean() + \
        #      ((flm_param1['pose'][:,:3] - flm_param2['pose'][:,:3])**2).mean()*1 + 1*((flm_param1['light'] - flm_param2['light'])**2).mean()
        para_loss = ((flm_param1['pose'][:,:3] - flm_param2['pose'][:,:3])**2).mean()
        #para_loss = ((flm_param1['pose'][:,3:] - flm_param2['pose'][:,3:])**2).mean() +  ((flm_param1['exp'] - flm_param2['exp'])**2).mean()
        flm_param1['pose'] = flm_param2['pose']
        flm_param1['exp'] = flm_param2['exp']
        flm_param1['light'] = flm_param2['light']
        #flm_param1['pose_flag'] = 0
       
        result2 = self.deca.decode(flm_param1)
        # flm_param1['pose'][:,:3]=0
        # result22 = self.deca.decode(flm_param1)
       
        # result2 = self.deca.decode(flm_param1,_pose_flag=1)
        loss = {}
        v_recon = (result1['vertices'] - result2['vertices'])**2
        recon_loss = (result1['rendered_images'] - result2['rendered_images'])**2
        #landmark_loss = (result1['landmarks3d'] - result2['landmarks3d'])**2
        
        # if preds is None:
        #     #preds = result1['landmarks2d']
        #     landmark_loss = torch.Tensor([0]).cuda()
        #     landmark_loss2 = torch.Tensor([0]).cuda()
        # else:
        landmark_loss = (result2['landmarks2d'][0] - result1['landmarks2d'][0]).to(deca_img1.device)**2
            # if flag == 0:
            #     landmark_loss = (result2['landmarks2d'][0] - result1['landmarks2d'][0]).to(deca_img1.device)**2
            # else:
            #     landmark_loss = (result2['landmarks2d'][0] - torch.Tensor(preds[0]).to(deca_img1.device))**2
            #landmark_loss2 = (result22['landmarks2d'][0] - result11['landmarks2d'][0]).to(deca_img1.device)**2
        #landmark_loss2 = torch.Tensor([0]).cuda()
        # print(landmark_loss.mean())
        # print(landmark_loss2.mean())

        

       
        loss['reconstruction'] = recon_loss
        loss['mesh'] = v_recon
        loss['landmark'] = landmark_loss
        loss['param'] = para_loss
        loss['rendered_image1'] = result1['rendered_images']
        loss['rendered_image2'] = result2['rendered_images']
        loss['flag'] = flag1 and flag2
        # print(result1['rendered_images'].max())
        #########################################################################################################################
        

        # input = io.imread('../test/assets/aflw-test.jpg')
        # print(deca_img1.shape)
        # exit(0)
        
        # print(preds)
        # exit(0)
        # point_size = 1
        # point_color = (0, 0, 255) 
        # thickness = 1 
        #points_list = (result1['landmarks2d']).detach().cpu().numpy()[0][19:]
    #     points_list = preds[0][18:]
    #     #points_list = [(160, 160), (136, 160), (150, 200), (200, 180), (120, 150), (145, 180)]
        
    #     # exit()
    #     img_v = img_v.copy()
    #    # print(img_v.dtype)
    #     #img_v = np.zeros((300, 300, 3), dtype="uint8") 
    #     #print(img_v.shape)
    #     for point in points_list:
    #         #print(point)
    #         cv2.circle(img_v, (int(point[0]),int(point[1])), point_size, point_color, thickness)

    #     # 画圆，圆心为：(160, 160)，半径为：60，颜色为：point_color，实心线
    #     #cv2.circle(img_v, (160, 160), 60, point_color, 0)

        
    #     Image.fromarray(img_v).save('1.jpg')
    #     exit()
        #print()
        ###########################################################################################################################
        # from PIL import Image
        # Image.fromarray(img_v).save('1.jpg')
        # exit()
        return loss

    def deca_loss_zeros(self,img1):
        
        deca_img1 = self.deca_align(self.face_detector, img1, self.deca_resolution_inp)
        flm_param1 = self.deca.encode(deca_img1)
        result1 = self.deca.decode(flm_param1)
        print(flm_param1['pose'])
        print(flm_param1['exp'])
        flm_param1['pose'] = flm_param1['pose']*0
        flm_param1['exp'] = flm_param1['exp']*0
        
        result_c = self.deca.decode(flm_param1)
        #result1 - result_c['exp']
        recon_loss = (result1['rendered_images'] - result_c['rendered_images'].detach())**2
        loss = {}
        # print(result1['rendered_images'].max(),result1['rendered_images'].min())
        # exit()
        #img_v = (result_c['rendered_images'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
        # img2 = (img2256.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
        # img3 = (interpolate(new_img_tensor2, (256, 256)).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
        # img4 = (interpolate(new_img_tensor3, (256, 256)).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
        # img5 = (interpolate(new_img_tensor4, (256, 256)).permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
                        
        # img_v = np.concatenate([img1, img2,img3,img4,img5], axis=1)
        #from PIL import Image
        #Image.fromarray(img_v).save('1.jpg')
        #exit()
        loss['reconstruction'] = recon_loss
       
        return loss

    def  extract_param(self,img):
        deca_img, _ = self.deca_align(self.face_detector, img, self.deca_resolution_inp)


        #from torchvision.utils import save_image
        #from PIL import Image
        # img1 = deca_img[0]
        # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
        #save_image(img1, 'img1.png')
        #exit()

        flm_param = self.deca.encode(deca_img)
        #result, vis = self.deca.decode(flm_param)
        #img = self.deca.visualize(vis)
        #img = Image.fromarray(img)
        #img.save('1.png')
        ##print(img.shape)
        # PIL()
        #exit()
        #flm_param['align_info'] = flag
        

        return flm_param

    def extract_neutral_obj(self,img,obj_name):
        deca_img, _ = self.deca_align(self.face_detector, img, self.deca_resolution_inp)
        flm_param = self.deca.encode(deca_img)
        flm_param['pose'] = flm_param['pose']*0
        flm_param['exp'] = flm_param['exp']*0
        result, vis = self.deca.decode(flm_param)
        self.deca.save_obj(obj_name,result)



    def extract_rendered(self,img,flag):
        
        deca_img, _ = self.deca_align1(self.face_detector, img, self.deca_resolution_inp)
    
        flm_param = self.deca.encode(deca_img)
        flm_param['pose_flag'] = flag
        result = self.deca.decode(flm_param)
        

        return result['rendered_images']

  # def extract_param_exp(self,)

    def grid_sample(self,image, optical):
        N, C, IH, IW = image.shape
        _, H, W, _ = optical.shape

        ix = optical[..., 0]
        iy = optical[..., 1]

        ix = ((ix + 1) / 2) * (IW-1)
        iy = ((iy + 1) / 2) * (IH-1)
        with torch.no_grad():
            ix_nw = torch.floor(ix)
            iy_nw = torch.floor(iy)
            ix_ne = ix_nw + 1
            iy_ne = iy_nw
            ix_sw = ix_nw
            iy_sw = iy_nw + 1
            ix_se = ix_nw + 1
            iy_se = iy_nw + 1

        nw = (ix_se - ix)    * (iy_se - iy)
        ne = (ix    - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix)    * (iy    - iy_ne)
        se = (ix    - ix_nw) * (iy    - iy_nw)
        
        with torch.no_grad():
            torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
            torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

            torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
            torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
    
            torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
            torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
    
            torch.clamp(ix_se, 0, IW-1, out=ix_se)
            torch.clamp(iy_se, 0, IH-1, out=iy_se)

        image = image.view(N, C, IH * IW)


        nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
        ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
        sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
        se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

        out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
                ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
                sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
                se_val.view(N, C, H, W) * se.view(N, 1, H, W))

        return out_val

def deca_align1(face_detector, img_tensor, msk_tensor=None, resolution_inp=224, scale=1.25):
    '''
    align a given img_tensor for DECA
    input:
        img_tensor: shape [1, C, H, W], range [-1, 1]
        resolution_inp: resolution of the output image tensor
    output:
        aligned_img_tensor: shape [1, C, resolution_inp, resolution_inp], range [0, 1]
    '''
    img = (img_tensor.detach()[0].permute(1, 2, 0).cpu().numpy() + 1) / 2. * 255
    bbox, bbox_type = face_detector.run(img)

    if len(bbox) < 4:
        print("No face detected!")
        exit()

    left = bbox[0]; right=bbox[2]
    top = bbox[1]; bottom=bbox[3]
    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
    size = int(old_size * scale)

    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
    src_pts = 2 * src_pts / img.shape[0] - 1
    DST_PTS = 2 * DST_PTS / resolution_inp - 1

    tform = estimate_transform('similarity', src_pts, DST_PTS)
    theta = torch.from_numpy(tform._inv_matrix[:2, :]).unsqueeze(0).cuda().float()
    grid = affine_grid(theta, (1, 3, resolution_inp, resolution_inp), align_corners=False)
    if msk_tensor is not None:
        img_tensor = torch.cat([img_tensor, msk_tensor], dim=1)
    img_inp = grid_sample(img_tensor, grid, align_corners=False)
    msk_inp = None
    if msk_tensor is not None:
        img_inp0 = img_inp[:, :-1, ...]
        msk_inp = img_inp[:, -1:, ...]
    else:
        img_inp0 = img_inp

    return 0.5 * (img_inp0 + 1), msk_inp

def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center
