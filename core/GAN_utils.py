"""
Native torch version of fc6 GANs. Support deployment on any machine, since the weights are publicly hosted online.
The motivation is to get rid of dependencies on Caffe framework totally.
"""
"""
Wrapper and loader of various GANs are listed, currently we have

* BigGAN
* BigBiGAN
* StyleGAN2
* PGGAN
* DCGAN
"""
#%%
import os
from os.path import join
from sys import platform
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from IPython.display import clear_output
from .build_montages import build_montages, color_framed_montages
from .torch_utils import progress_bar
load_urls = False
if platform == "linux":  # CHPC cluster
    # homedir = os.path.expanduser('~')
    # netsdir = os.path.join(homedir, 'Generate_DB/nets')
    homedir = "/scratch/binxu"
    netsdir = "/scratch/binxu/torch/checkpoints"
    load_urls = True # note it will try to load from $TORCH_HOME\checkpoints\"upconvGAN_%s.pt"%"fc6"
    # ckpt_path = {"vgg16": "/scratch/binxu/torch/vgg16-397923af.pth"}
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        homedir = "D:/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2C':  # PonceLab-Desktop Victoria
        homedir = r"C:\Users\ponce\Documents\Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B':
        homedir = r"C:\Users\Ponce lab\Documents\Python\Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A':
        homedir = r"C:\Users\Poncelab-ML2a\Documents\Python\Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        homedir = "E:/Monkey_Data/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  # Home_WorkStation Victoria
        homedir = "C:/Users/zhanq/OneDrive - Washington University in St. Louis/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    else:
        load_urls = True
        homedir = os.path.expanduser('~')
        netsdir = os.path.join(homedir, 'Documents/nets')

model_urls = {"pool5" : "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145337&authkey=AFaUAgeoIg0WtmA",
    "fc6": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145339&authkey=AC2rQMt7Obr0Ba4",
    "fc7": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145338&authkey=AJ0R-daUAVYjQIw",
    "fc8": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145340&authkey=AKIfNk7s5MGrRkU"}

def load_statedict_from_online(name="fc6"):
    torchhome = torch.hub._get_torch_home()
    ckpthome = join(torchhome, "checkpoints")
    os.makedirs(ckpthome, exist_ok=True)
    filepath = join(ckpthome, "upconvGAN_%s.pt"%name)
    if not os.path.exists(filepath):
        torch.hub.download_url_to_file(model_urls[name], filepath, hash_prefix=None,
                                   progress=True)
    SD = torch.load(filepath)
    return SD

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

RGB_mean = torch.tensor([123.0, 117.0, 104.0])
RGB_mean = torch.reshape(RGB_mean, (1, 3, 1, 1))

class upconvGAN(nn.Module):
    def __init__(self, name="fc6", pretrained=True, shuffled=True):
        super(upconvGAN, self).__init__()
        self.name = name
        if name == "fc6" or name == "fc7":
            self.G = nn.Sequential(OrderedDict([
        ('defc7', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc6', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc5', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('reshape', View((-1, 256, 4, 4))),
        ('deconv5', nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
            ]))
            self.codelen = self.G[0].in_features
        elif name == "fc8":
            self.G = nn.Sequential(OrderedDict([
  ("defc7", nn.Linear(in_features=1000, out_features=4096, bias=True)),
  ("relu_defc7", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("defc6", nn.Linear(in_features=4096, out_features=4096, bias=True)),
  ("relu_defc6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("defc5", nn.Linear(in_features=4096, out_features=4096, bias=True)),
  ("relu_defc5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("reshape", View((-1, 256, 4, 4))),
  ("deconv5", nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv5_1", nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv5_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv4", nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv4_1", nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv4_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv3", nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv3_1", nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv3_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv2", nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv1", nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv0", nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ]))
            self.codelen = self.G[0].in_features
        elif name == "pool5":
            self.G = nn.Sequential(OrderedDict([
        ('Rconv6', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv7', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv8', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))),
        ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv5', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ]))
            self.codelen = self.G[0].in_channels
        # load pre-trained weight from online or local folders
        if pretrained:
            if load_urls:
                SDnew = load_statedict_from_online(name)
            else:
                savepath = {"fc6": join(netsdir, r"upconv/fc6/generator_state_dict.pt"),
                            "fc7": join(netsdir, r"upconv/fc7/generator_state_dict.pt"),
                            "fc8": join(netsdir, r"upconv/fc8/generator_state_dict.pt"),
                            "pool5": join(netsdir, r"upconv/pool5/generator_state_dict.pt")}
                SD = torch.load(savepath[name])
                SDnew = OrderedDict()
                for name, W in SD.items():  # discard this inconsistency
                    name = name.replace(".1.", ".")
                    SDnew[name] = W
            self.G.load_state_dict(SDnew)

    def forward(self, x):
        return self.G(x)[:, [2, 1, 0], :, :]

    def visualize(self, x, scale=1.0):
        raw = self.G(x)
        return torch.clamp(raw[:, [2, 1, 0], :, :] + RGB_mean.to(raw.device), 0, 255.0) / 255.0 * scale

    def render(self, x, scale=1.0, B=42):  # add batch processing to avoid memory over flow for batch too large
        coden = x.shape[0]
        img_all = []
        csr = 0  # if really want efficiency, we should use minibatch processing.
        while csr < coden:
            csr_end = min(csr + B, coden)
            with torch.no_grad():
                imgs = self.visualize(torch.from_numpy(x[csr:csr_end, :]).float().cuda(), scale).permute(2,3,1,0).cpu().numpy()
            img_all.extend([imgs[:, :, :, imgi] for imgi in range(imgs.shape[3])])
            csr = csr_end
        return img_all

    def visualize_batch_np(self, codes_all_arr, scale=1.0, B=42):
        coden = codes_all_arr.shape[0]
        img_all = None
        csr = 0  # if really want efficiency, we should use minibatch processing.
        with torch.no_grad():
            while csr < coden:
                csr_end = min(csr + B, coden)
                imgs = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(), scale).cpu()
                img_all = imgs if img_all is None else torch.cat((img_all, imgs), dim=0)
                csr = csr_end
        return img_all


def visualize_np(G, code, layout=None, show=True):
    """Utility function to visualize a np code vectors.

    If it's a single vector it will show in a plt window, Or it will show a montage in a windows photo.
    G: a generator equipped with a visualize method to turn torch code into torch images.
    layout: controls the layout of the montage. (5,6) create 5 by 6 grid
    show: if False, it will return the images in 4d array only.
    """
    with torch.no_grad():
        imgs = G.visualize(torch.from_numpy(code).float().cuda()).cpu().permute([2, 3, 1, 0]).squeeze().numpy()
    if show:
        if len(imgs.shape) <4:
            plt.imshow(imgs)
            plt.show()
        else:
            img_list = [imgs[:,:,:,imgi].squeeze() for imgi in range(imgs.shape[3])]
            if layout is None:
                mtg = build_montages(img_list,(256,256),(imgs.shape[3],1))[0]
                Image.fromarray(np.uint8(mtg*255.0)).show()
            else:
                mtg = build_montages(img_list, (256, 256), layout)[0]
                Image.fromarray(np.uint8(mtg*255.0)).show()
    return imgs

#%% Various GAN wrappers.
#%% BigGAN wrapper 
def loadBigGAN(version="biggan-deep-256"):
    from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, BigGANConfig
    if platform == "linux":
        cache_path = "/scratch/binxu/torch/"
        cfg = BigGANConfig.from_json_file(join(cache_path, "%s-config.json" % version))
        BGAN = BigGAN(cfg)
        BGAN.load_state_dict(torch.load(join(cache_path, "%s-pytorch_model.bin" % version)))
    else:
        BGAN = BigGAN.from_pretrained(version)
    for param in BGAN.parameters():
        param.requires_grad_(False)
    # embed_mat = BGAN.embeddings.parameters().__next__().data
    BGAN.cuda()
    return BGAN

class BigGAN_wrapper():#nn.Module
    def __init__(self, BigGAN, space="class"):
        self.BigGAN = BigGAN
        self.space = space

    def sample_vector(self, sampn=1, class_id=None, device="cuda"):
        if class_id is None:
            refvec = torch.cat((0.7 * torch.randn(128, sampn).to(device),
                                self.BigGAN.embeddings.weight[:, torch.randint(1000, size=(sampn,))].to(device),)).T
        else:
            refvec = torch.cat((0.7 * torch.randn(128, sampn).to(device),
                                self.BigGAN.embeddings.weight[:, (class_id*torch.ones(sampn)).long()].to(device),)).T
        return refvec

    def visualize(self, code, scale=1.0, truncation=0.7):
        imgs = self.BigGAN.generator(code, truncation)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch(self, codes_all, truncation=0.7, B=15):
        csr = 0
        img_all = None
        imgn = codes_all.shape[0]
        with torch.no_grad():
            while csr < imgn:
                csr_end = min(csr + B, imgn)
                img_list = self.visualize(codes_all[csr:csr_end, :].cuda(), truncation=truncation, ).cpu()
                img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
                csr = csr_end
        return img_all

    def visualize_batch_np(self, codes_all_arr, truncation=0.7, B=15):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        with torch.no_grad():
            while csr < imgn:
                csr_end = min(csr + B, imgn)
                code_batch = torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda()
                img_list = self.visualize(code_batch, truncation=truncation, ).cpu()
                img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
                csr = csr_end
                clear_output(wait=True)
                progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

    def render(self, codes_all_arr, truncation=0.7, B=15):
        img_tsr = self.visualize_batch_np(codes_all_arr, truncation=truncation, B=B)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]
#%%
import sys
if platform == "linux":
    BigBiGAN_root = r"/home/binxu/BigGANsAreWatching"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        BigBiGAN_root = r"D:\Github\BigGANsAreWatching"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        BigBiGAN_root = r"E:\Github_Projects\BigGANsAreWatching"
    else:
        BigBiGAN_root = r"D:\Github\BigGANsAreWatching"
# the model is on cuda from this.
def loadBigBiGAN(weightpath=None):
    sys.path.append(BigBiGAN_root)
    from BigGAN.gan_load import UnconditionalBigGAN, make_big_gan
    # from BigGAN.model.BigGAN import Generator
    if weightpath is None:
        weightpath = join(BigBiGAN_root, "BigGAN\weights\BigBiGAN_x1.pth")
    BBGAN = make_big_gan(weightpath, resolution=128)
    # BBGAN = make_big_gan(r"E:\Github_Projects\BigGANsAreWatching\BigGAN\weights\BigBiGAN_x1.pth", resolution=128)
    for param in BBGAN.parameters():
        param.requires_grad_(False)
    BBGAN.eval()
    return BBGAN
#%%
class BigBiGAN_wrapper():#nn.Module
    def __init__(self, BigBiGAN, ):
        self.BigGAN = BigBiGAN

    def sample_vector(self, sampn=1, device="cuda"):
        refvec = torch.randn((sampn, 120)).to(device)
        return refvec

    def visualize(self, code, scale=1.0, resolution=256):
        imgs = self.BigGAN(code, )
        imgs = F.interpolate(imgs, size=(resolution, resolution), align_corners=True, mode='bilinear')
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, B=15, scale=1.0, resolution=256):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        with torch.no_grad():
            while csr < imgn:
                csr_end = min(csr + B, imgn)
                code_batch = torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda()
                img_list = self.visualize(code_batch, scale=scale, resolution=resolution).cpu()
                img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
                csr = csr_end
                # clear_output(wait=True)
                # progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

    def render(self, codes_all_arr, B=15, resolution=256, scale=1.0, ):
        img_tsr = self.visualize_batch_np(codes_all_arr, scale=scale, resolution=resolution, B=B)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]    

    # def render(self, codes_all_arr, B=15, scale=1.0, resolution=256):
    #     img_tsr = None
    #     imgn = codes_all_arr.shape[0]
    #     csr = 0
    #     with torch.no_grad():
    #         while csr < imgn:
    #             csr_end = min(csr + B, imgn)
    #             code_batch = torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda()
    #             img_list = self.visualize(code_batch, scale=scale, resolution=resolution).cpu()
    #             img_tsr = img_list if img_tsr is None else torch.cat((img_tsr, img_list), dim=0)
    #             csr = csr_end
    #     return [img.permute([1,2,0]).numpy() for img in img_tsr]
#%% StyleGAN2 wrapper 
import sys
if platform == "linux":  # CHPC cluster
    StyleGAN2_root = "./stylegan2-pytorch"
    ckpt_root = "./pretrained_models"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        StyleGAN2_root = r"D:\Github\stylegan2-pytorch"
        ckpt_root = join(StyleGAN2_root, 'checkpoint')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        StyleGAN2_root = r"E:\DL_Projects\Vision\stylegan2-pytorch"
        ckpt_root = join(StyleGAN2_root, 'checkpoint')
    else:
        StyleGAN2_root = r"E:\DL_Projects\Vision\stylegan2-pytorch"
        ckpt_root = join(StyleGAN2_root, 'checkpoint')


def loadStyleGAN2(ckpt_name="stylegan2-ffhq-config-f.pt", channel_multiplier=2, n_mlp=8, latent=512, size=512,
                  device="cpu"):
    sys.path.append(StyleGAN2_root)
    configtab = {"stylegan2-cat-config-f.pt": (256, 2),
                 "ffhq-256-config-e-003810.pt": (256, 1),
                 "ffhq-512-avg-tpurun1.pt": (512, 2),
                 "stylegan2-ffhq-config-f.pt": (1024, 2),
                 "2020-01-11-skylion-stylegan2-animeportraits.pt": (512, 2),
                 "stylegan2-car-config-f.pt": (512, 2),
                 "model.ckpt-533504.pt": (512, 2)}
    from model import Generator
    ckpt_path = join(ckpt_root, ckpt_name)
    try:
        size, channel_multiplier = configtab[ckpt_name]
        print("Checkpoint name found, use config from memory.\nsize %d chan mult %d n mlp %d latent %d"%
              (size, channel_multiplier, n_mlp, latent))
    except KeyError:
        print("Checkpoint name not found, use config from input.\nsize %d chan mult %d n mlp %d latent %d"%
              (size, channel_multiplier, n_mlp, latent))
    g_ema = Generator(
        size, latent, n_mlp, channel_multiplier=channel_multiplier
    ).to(device)
    try:
        checkpoint = torch.load(ckpt_path)
    except:
        print("Checkpoint %s load failed, Available Checkpoints: "%ckpt_name, os.listdir(ckpt_path))
    g_ema.load_state_dict(checkpoint['g_ema'])
    g_ema.eval()
    for param in g_ema.parameters():
        param.requires_grad_(False)
    g_ema.cuda()
    return g_ema
#%%
class StyleGAN2_wrapper():#nn.Module
    def __init__(self, StyleGAN, ):
        self.StyleGAN = StyleGAN
        truncation = 0.8  # Note these parameters could be tuned
        truncation_mean = 4096
        mean_latent = StyleGAN.mean_latent(truncation_mean)
        self.truncation = truncation
        self.mean_latent = mean_latent
        self.wspace = False
        self.random = True

    def sample_vector(self, sampn=1, device="cuda"):
        if not self.wspace:
            refvec = torch.randn((sampn, 512)).to(device)
        else:
            refvec_Z = torch.randn((sampn, 512)).to(device)
            refvec = self.StyleGAN.style(refvec_Z).to(device)
        return refvec

    def select_trunc(self, truncation, truncation_mean=4096):
        self.truncation = truncation
        mean_latent = self.StyleGAN.mean_latent(truncation_mean)
        self.mean_latent = mean_latent


    def fix_noise(self, random=False):
        self.random = False
        return self.StyleGAN.noise

    def use_wspace(self, wspace=True):
        self.wspace = wspace

    def visualize(self, code, scale=1.0, resolution=256, truncation=None, mean_latent=None, wspace=None):
        if truncation is None:  truncation = self.truncation
        if mean_latent is None:  mean_latent = self.mean_latent
        if wspace is None:  wspace = self.wspace
        if truncation is None: truncation=0.75
        # print(code.shape)
        
        # exit()
        # print(code.requires_grad)
        # print('code',code.shape)
        # exit()
        
        code = code.reshape(-1,18,512)
        new_s1 = code[:,:3].clone().detach()
        new_s2 = code[:,3:10].clone()#.detach()
        new_s3 = code[:,10:].clone().detach()
        
        code = torch.cat((new_s1,new_s2,new_s3),dim=1)
        # code[:,6] = code[:,6]
        imgs, _ = self.StyleGAN([code], input_is_latent=wspace, randomize_noise=self.random)
        imgs = F.interpolate(imgs, size=(resolution, resolution), align_corners=True, mode='bilinear')
        # print(imgs.requires_grad)
        # exit()
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, truncation=None, mean_latent=None, B=15):
        if truncation is None:  truncation = self.truncation
        if mean_latent is None:  mean_latent = self.mean_latent
        if self.StyleGAN.size == 1024:  B = round(B/4)
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            with torch.no_grad():
                img_list = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(),
                                       truncation=truncation, mean_latent=mean_latent, ).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            csr = csr_end
            clear_output(wait=True)
            progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

    def render(self, codes_all_arr, truncation=None, mean_latent=None, B=15):
        if truncation is None:  truncation = self.truncation
        if mean_latent is None:  mean_latent = self.mean_latent
        img_tsr = self.visualize_batch_np(codes_all_arr, truncation=truncation, mean_latent=mean_latent, B=B)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]


#%%
import math
if platform == "linux":  # CHPC cluster
    StyleGAN1_root = r"/home/binxu/stylegan2-pytorch"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        StyleGAN1_root = r"D:\Github\style-based-gan-pytorch"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        StyleGAN1_root = r"E:\Github_Projects\style-based-gan-pytorch"
    else:
        StyleGAN1_root = r"E:\Github_Projects\style-based-gan-pytorch"

def loadStyleGAN():
    sys.path.append(StyleGAN1_root)
    ckpt_root = join(StyleGAN1_root, 'checkpoint')
    from model import StyledGenerator
    from generate import get_mean_style
    import math
    generator = StyledGenerator(512).to("cuda")
    # generator.load_state_dict(torch.load(r"E:\Github_Projects\style-based-gan-pytorch\checkpoint\stylegan-256px-new.model")['g_running'])
    generator.load_state_dict(torch.load(join(StyleGAN1_root, "checkpoint\stylegan-256px-new.model"))[
                                  'g_running'])
    generator.eval()
    for param in generator.parameters():
        param.requires_grad_(False)
    return generator

class StyleGAN_wrapper():  # nn.Module
    def __init__(self, StyleGAN, resolution=256):
        sys.path.append(StyleGAN1_root)
        from generate import get_mean_style
        self.StyleGAN = StyleGAN
        self.mean_style = get_mean_style(StyleGAN, "cuda")  # note this is a stochastic process so can differ from
                                                            # init to init.
        self.step = int(math.log(resolution, 2)) - 2
        self.wspace = False
        self.random = True

    def fix_noise(self, noise=None):
        self.random = False
        if noise is None: noise = [torch.randn(1, 1, 4 * 2 ** i, 4 * 2 ** i, device="cuda") for i in range(self.step + 1)]
        self.fixed_noise = noise
        return self.fixed_noise

    def use_wspace(self, wspace=True):
        self.wspace = wspace

    def visualize(self, code, scale=1.0, resolution=256, mean_style=None, wspace=False, noise=None):
        # if step is None: step = self.step
        step = int(math.log(resolution, 2)) - 2
        if not self.random:
            noise = [noise_l.repeat(code.shape[0], 1, 1, 1) for noise_l in self.fixed_noise]
        elif self.random and noise is None:
            noise = [torch.randn(code.shape[0], 1, 4 * 2 ** i, 4 * 2 ** i, device="cuda") for i in range(step + 1)]
        if not wspace and not self.wspace:
            if mean_style is None: mean_style = self.mean_style
            imgs = self.StyleGAN(code, noise=noise, step=step, alpha=1,
                mean_style=mean_style, style_weight=0.7,
            )
        else: # code ~ 0.2 * torch.randn(1, 1, 512)
            imgs = self.StyleGAN.generator([code], noise, step=step)
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, resolution=256, mean_style=None, B=15, wspace=False, noise=None):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            with torch.no_grad():
                img_list = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(),
                                       resolution=resolution, mean_style=mean_style, wspace=wspace, noise=noise).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            csr = csr_end
            # clear_output(wait=True)
            # progress_bar(csr_end, imgn, "ploting row of page: %d of %d" % (csr_end, imgn))
        return img_all

    def render(self, codes_all_arr, resolution=256, mean_style=None, B=15, wspace=False, noise=None):
        img_tsr = self.visualize_batch_np(codes_all_arr, resolution=resolution, mean_style=mean_style, B=B,
                                          wspace=wspace, noise=noise)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]
# G = StyleGAN_wrapper(generator)
#%% PGGAN load 
def loadPGGAN(onlyG=True): 
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-256',
                       pretrained=True, useGPU=True)
    if onlyG:
        return model.avgG
    else:
        return model 

class PGGAN_wrapper():  # nn.Module
    """
    model = loadPGGAN(onlyG=False)
    G = PGGAN_wrapper(model.avgG)

    model = loadPGGAN()
    G = PGGAN_wrapper(model)
    """
    def __init__(self, PGGAN, ):
        self.PGGAN = PGGAN

    def sample_vector(self, sampn=1, device="cuda"):
        refvec = torch.randn((sampn, 512)).to(device)
        return refvec

    def visualize(self, code, scale=1.0):
        imgs = self.PGGAN.forward(code,)
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, scale=1.0, B=50):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            with torch.no_grad():
                img_list = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(),
                        scale=scale).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            csr = csr_end
        return img_all

    def render(self, codes_all_arr, scale=1.0, B=50):
        img_tsr = self.visualize_batch_np(codes_all_arr, scale=scale, B=B)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]
# G = PGGAN_wrapper(model.avgG)


#%% DCGAN load 
def loadDCGAN(onlyG=True): 
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 
            'DCGAN', pretrained=True, useGPU=True)
    if onlyG:
        return model.avgG
    else:
        return model 

class DCGAN_wrapper():  # nn.Module
    def __init__(self, DCGAN, ):
        self.DCGAN = DCGAN

    def sample_vector(self, sampn=1, device="cuda"):
        refvec = torch.randn((sampn, 120)).to(device)
        return refvec

    def visualize(self, code, scale=1.0):
        imgs = self.DCGAN(code,)  # Matlab version default to 0.7
        return torch.clamp((imgs + 1.0) / 2.0, 0, 1) * scale

    def visualize_batch_np(self, codes_all_arr, scale=1.0, B=50):
        csr = 0
        img_all = None
        imgn = codes_all_arr.shape[0]
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            with torch.no_grad():
                img_list = self.visualize(torch.from_numpy(codes_all_arr[csr:csr_end, :]).float().cuda(),
                        scale=scale).cpu()
            img_all = img_list if img_all is None else torch.cat((img_all, img_list), dim=0)
            csr = csr_end
        return img_all

    def render(self, codes_all_arr, scale=1.0, B=50):
        img_tsr = self.visualize_batch_np(codes_all_arr, scale=scale, B=B)
        return [img.permute([1,2,0]).numpy() for img in img_tsr]

#%% The first time to run this you need these modules
if __name__ == "__main__":
    import sys
    import matplotlib.pylab as plt
    sys.path.append(r"E:\Github_Projects\Visual_Neuro_InSilico_Exp")
    from torch_net_utils import load_generator, visualize
    G = load_generator(GAN="fc6")
    UCG = upconvGAN("fc6")
    #%%
    def test_consisitency(G, UCG):#_
        code = torch.randn((1, UCG.codelen))
        # network outputs are the same.
        assert torch.allclose(UCG(code), G(code)['deconv0'][:, [2, 1, 0], :, :])
        # visualization function is the same
        imgnew = UCG.visualize(code).permute([2, 3, 1, 0]).squeeze()
        imgorig = visualize(G, code.numpy(), mode="cpu")
        assert torch.allclose(imgnew, imgorig)
        plt.figure(figsize=[6,3])
        plt.subplot(121)
        plt.imshow(imgnew.detach())
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(imgorig.detach())
        plt.axis('off')
        plt.show()
    test_consisitency(G, UCG)
    #%%
    G = load_generator(GAN="fc7")
    UCG = upconvGAN("fc7")
    test_consisitency(G, UCG)
    #%%
    # pool5 GAN
    def test_FCconsisitency(G, UCG):#_
        code = torch.randn((1, UCG.codelen, 6, 6))
        # network outputs are the same.
        assert torch.allclose(UCG(code), G(code)['generated'][:, [2, 1, 0], :, :])
        # visualization function is the same
        imgnew = UCG.visualize(code).permute([2, 3, 1, 0]).squeeze()
        imgorig = G(code)['generated'][:, [2, 1, 0], :, :]
        imgorig = torch.clamp(imgorig + RGB_mean, 0, 255.0) / 255.0
        imgorig = imgorig.permute([2, 3, 1, 0]).squeeze()
        # imgorig = visualize(G, code.numpy(), mode="cpu")
        # assert torch.allclose(imgnew, imgorig)
        plt.figure(figsize=[6,3])
        plt.subplot(121)
        plt.imshow(imgnew.detach())
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(imgorig.detach())
        plt.axis('off')
        plt.show()
    G = load_generator(GAN="pool5")
    UCG = upconvGAN("pool5")
    test_FCconsisitency(G, UCG)

#%% This can work~
# G = upconvGAN("pool5")
# G.G.load_state_dict(torch.hub.load_state_dict_from_url(r"https://drive.google.com/uc?export=download&id=1vB_tOoXL064v9D6AKwl0gTs1a7jo68y7",progress=True))
