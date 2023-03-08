# coding: utf-8

__author__ = 'cleardusk'

import sys
sys.path.append('TDDFA')
import argparse
import cv2
import yaml
from PIL import Image
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from torch.nn import CosineSimilarity
# from utils.render import render
# #from utils.render_ctypes import render  # faster
# from utils.depth import depth
# from utils.pncc import pncc
# from utils.uv import uv_tex
from utils.pose import viz_pose, estimate_angle
# from utils.serialization import ser_to_ply, ser_to_obj
# from utils.functions import draw_landmarks, get_suffix
# from utils.tddfa_util import str2bool
import torchvision.transforms as transforms
import torch

class Estimate_pose(object):
    def __init__(self):
        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        # Init FaceBoxes and TDDFA, recommend using onnx flag
        # if args.onnx:
        #     import os
        #     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        #     os.environ['OMP_NUM_THREADS'] = '4'

        #     from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        #     from TDDFA_ONNX import TDDFA_ONNX

        #     face_boxes = FaceBoxes_ONNX()
        #     tddfa = TDDFA_ONNX(**cfg)
        # else:
        gpu_mode = True
        self.tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        self.face_boxes = FaceBoxes()

        #resolution =  args.get('size', 120)
        self.transform = transforms.Compose([
                #transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def estimate_pose(self,img):
        # img = img.squeeze()
        #print(img.shape)
        box_list = []
        for img_s in img:
            img_s = (img_s.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[...,::-1]
            #print(img.max())

            

            #Detect faces, get 3DMM params and roi boxes
            boxes = self.face_boxes(img_s)
            n = len(boxes)
            if n == 0:
                print(f'No face detected, exit')
                sys.exit(-1)
            else:
                box = boxes[0]
                box_list.append(box)
            #print(f'Detect {n} faces')

        #real_image = Image.open(args.img_fp)
        img_t =  img
        #print('pose:',img_t.shape)
        param_lst, roi_box_lst = self.tddfa(img_t.flip(1),box_list)
        # Visualization and serialization
        # dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
        # old_suffix = get_suffix(args.img_fp)
        # new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

        # wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

        #ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        #print('pose:',param_lst.shape)
        pose = estimate_angle(param_lst)
        # print(param_lst.shape)
        # exit()
        #print('pose:',pose.shape)
        # exit()
        return pose
        #return param_lst[:,52:]

    def pose_loss(self,img1,img2):
        #print(img1.shape)
        # img1 = img1.squeeze()
        # img2 = img2.squeeze()
        # print('a',img1.shape)
        # print('b',img2.shape)
        #return ((img2-img1)**2).mean()
        return torch.cos(self.estimate_pose(img1)[:,2] - self.estimate_pose(img2)[:,2])
        #return ((self.estimate_pose(img1) - self.estimate_pose(img2))**2).mean(dim=1)


def test():
    ep = Estimate_pose()
    real_image1 = Image.open('/home/jdq/face-swap/training_set/backgrounds/00335.png')
    real_image2 = Image.open('/home/jdq/face-swap/training_set/backgrounds/00134.png')
    img1 =  ep.transform(real_image1)
    img2 =  ep.transform(real_image2)
    alpha = ep.pose_loss(img1,img2)
    print(alpha)



def main(args):
    ####################################################################
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    #resolution =  args.get('size', 120)
    transform = transforms.Compose([
            #transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    #####################################################################

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)
    print(img.max())

    

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    real_image = Image.open(args.img_fp)
    img_t =  transform(real_image)

    param_lst, roi_box_lst = tddfa(img_t, boxes)

    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '2d_dense':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '3d':
        render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'depth':
        # if `with_bf_flag` is False, the background is black
        depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'pncc':
        pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'uv_tex':
        uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'pose':
        viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'ply':
        ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    elif args.opt == 'obj':
        ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    else:
        raise ValueError(f'Unknown opt {args.opt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='pose',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    test()
    #main(args)
