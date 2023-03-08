"""Utils to load the computed Hessian information. Data management."""
import torch
import numpy as np
import os, os.path, sys
from os.path import join
if sys.platform == "linux":
    summarydir = r"/scratch/binxu/Hessian_summary"
else:
    summarydir = r"E:\OneDrive - Washington University in St. Louis\Hessian_summary"

def load_Hcorrmat(GAN, spec=None, nandiag=True):
    matchstr = GAN
    if matchstr not in corrmat_npz_dict:
        print(list(corrmat_npz_dict), " Choose from these.")
    Hccpath = join(summarydir, corrmat_npz_dict[matchstr])
    with np.load(Hccpath) as data:
        print(list(data))
        corr_mat_log, corr_mat_lin, = data['corr_mat_log'], data['corr_mat_lin']
        corr_mat_log_nodiag, corr_mat_lin_nodiag = corr_mat_log.copy(), corr_mat_lin.copy()
        np.fill_diagonal(corr_mat_log_nodiag, np.nan)
        np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
        cclogmean, cclogstd = np.nanmean(corr_mat_log), np.nanstd(corr_mat_log)
        cclinmean, cclinstd = np.nanmean(corr_mat_lin), np.nanstd(corr_mat_lin)
        print("log scale corr %.3f(%.3f), lin scale corr %.3f(%.3f)" % (cclogmean, cclogstd, cclinmean, cclinstd))
        if nandiag:
            return corr_mat_log_nodiag, corr_mat_lin_nodiag
        else:
            return corr_mat_log, corr_mat_lin

def load_Haverage(GAN, spec=None, descend=None, abssort=True):
    if os.path.exists(GAN):
        with np.load(GAN) as data:
            H, eva, evc = data["H_avg"], data["eva_avg"], data["evc_avg"]
    else:
        # if spec is None:
        matchstr = GAN
        if matchstr not in Havg_npz_dict:
            print(list(Havg_npz_dict), " Choose from these.")
        Hpath = join(summarydir, Havg_npz_dict[matchstr])
        with np.load(Hpath) as data:
            print(list(data))
            if GAN == "BigGAN":
                if spec == "class":
                    H, eva, evc = data["H_clas_avg"], data["eigvals_clas_avg"], data["eigvects_clas_avg"]
                elif spec == "noise":
                    H, eva, evc = data["H_nois_avg"], data["eigvals_nois_avg"], data["eigvects_nois_avg"]
                else:
                    H, eva, evc = data["H_avg"], data["eigvals_avg"], data["eigvects_avg"]
            elif GAN == "fc6GAN":
                H, eva, evc = data['H_avg'], data['eigv_avg'], data['eigvect_avg']
            else:
                H, eva, evc = data["H_avg"], data["eva_avg"], data["evc_avg"]
    if descend is None: # if not descent then ascend by default. 
        return H, eva, evc
    if abssort:
        eva = np.abs(eva)
    if descend:
        sort_idx = np.argsort(-eva)
    else:
        sort_idx = np.argsort(eva)
    eva = eva[sort_idx].copy()
    evc = evc[:, sort_idx].copy()
    return H, eva, evc

Havg_npz_dict = {"fc6GAN": "fc6GAN/Evolution_Avg_Hess.npz",
     "DCGAN": "DCGAN/H_avg_DCGAN.npz", 
     "BigGAN": "BigGAN/H_avg_1000cls.npz",
    "BigGAN_noise": "BigGAN/H_avg_1000cls.npz",
    "BigGAN_class": "BigGAN/H_avg_1000cls.npz",
     "BigBiGAN": "BigBiGAN/H_avg_BigBiGAN.npz",
     "PGGAN": "PGGAN/H_avg_PGGAN.npz",
     "StyleGAN-Face*": "StyleGAN/H_avg_StyleGAN.npz",
     "StyleGAN2-Face512*": "StyleGAN2/H_avg_ffhq-512-avg-tpurun1.npz",
     "StyleGAN2-Face256*": "StyleGAN2/H_avg_ffhq-256-config-e-003810.npz",
     "StyleGAN2-Cat256*": "StyleGAN2/H_avg_stylegan2-cat-config-f.npz",
     "StyleGAN2-Face1024*": "StyleGAN2_Fix/stylegan2-face1024-noise/H_avg_StyleGAN2-Face1024_Z.npz",
      "StyleGAN-Face_Z": "StyleGAN_Fix/StyleGAN_Face256_fix/H_avg_StyleGAN_Face256_fix.npz",
      "StyleGAN2-Face512_Z": "StyleGAN2_Fix/ffhq-512-avg-tpurun1_fix/H_avg_ffhq-512-avg-tpurun1_fix.npz",
      "StyleGAN2-Face256_Z": "StyleGAN2_Fix/ffhq-256-config-e-003810_fix/H_avg_ffhq-256-config-e-003810_fix.npz",
      "StyleGAN2-Cat256_Z": "StyleGAN2_Fix/stylegan2-cat-config-f_fix/H_avg_stylegan2-cat-config-f_fix.npz",
      "StyleGAN2-Face1024_Z": "StyleGAN2_Fix/stylegan2-ffhq-config-f_fix/H_avg_StyleGAN_FFHQ1024.npz",
      "StyleGAN-Face_W": "StyleGAN_Fix/StyleGAN_Face256_W_fix/H_avg_StyleGAN_Face256_W_fix.npz",
      "StyleGAN2-Face512_W": "StyleGAN2_Fix/ffhq-512-avg-tpurun1_W_fix/H_avg_ffhq-512-avg-tpurun1_W_fix.npz",
      "StyleGAN2-Face256_W": "StyleGAN2_Fix/ffhq-256-config-e-003810_W_fix/H_avg_ffhq-256-config-e-003810_W_fix.npz",
      "StyleGAN2-Cat256_W": "StyleGAN2_Fix/stylegan2-cat-config-f_W_fix/H_avg_stylegan2-cat-config-f_W_fix.npz",
      "WaveGAN_piano_MSE":  "WaveGAN/H_avg_WaveGAN_piano_MSE.npz",
      "WaveGAN_piano_STFT": "WaveGAN/H_avg_WaveGAN_piano_STFT.npz",
                    }

spectra_npz_dict = {"fc6GAN": "FC6GAN/spectra_col_evol.npz",
          "DCGAN": "DCGAN/spectra_col_BP.npz",
          "BigGAN": "BigGAN/spectra_col.npz",
          "BigGAN_noise": "BigGAN/spectra_col.npz",
          "BigGAN_class": "BigGAN/spectra_col.npz",
          "BigBiGAN": "BigBiGAN/spectra_col.npz",
          "PGGAN": "PGGAN/spectra_col_BP.npz",
          "StyleGAN-Face*": "StyleGAN/spectra_col_face256_BP.npz",
          "StyleGAN2-Face512*": "StyleGAN2/spectra_col_FFHQ512.npz",
          "StyleGAN2-Face256*": "StyleGAN2/spectra_col_ffhq-256-config-e-003810_BP.npz",
          "StyleGAN2-Cat256*": "StyleGAN2/spectra_col_stylegan2-cat-config-f_BP.npz", 
          "StyleGAN-Face_Z": "StyleGAN_Fix/StyleGAN_Face256_fix/spectra_col_StyleGAN_Face256_fix.npz", 
          "StyleGAN2-Face512_Z": "StyleGAN2_Fix/ffhq-512-avg-tpurun1_fix/spectra_col_ffhq-512-avg-tpurun1_fix.npz", 
          "StyleGAN2-Face256_Z": "StyleGAN2_Fix/ffhq-256-config-e-003810_fix/spectra_col_ffhq-256-config-e-003810_fix.npz", 
          "StyleGAN2-Cat256_Z": "StyleGAN2_Fix/stylegan2-cat-config-f_fix/spectra_col_stylegan2-cat-config-f_fix.npz",
          "StyleGAN2-Face1024_Z": "StyleGAN2_Fix/stylegan2-ffhq-config-f_fix/spectra_col_StyleGAN_FFHQ1024.npz",
          "StyleGAN-Face_W": "StyleGAN_Fix/StyleGAN_Face256_W_fix/spectra_col_StyleGAN_Face256_W_fix.npz", 
          "StyleGAN2-Face512_W": "StyleGAN2_Fix/ffhq-512-avg-tpurun1_W_fix/spectra_col_ffhq-512-avg-tpurun1_W_fix.npz", 
          "StyleGAN2-Face256_W": "StyleGAN2_Fix/ffhq-256-config-e-003810_W_fix/spectra_col_ffhq-256-config-e-003810_W_fix.npz", 
          "StyleGAN2-Cat256_W": "StyleGAN2_Fix/stylegan2-cat-config-f_W_fix/spectra_col_stylegan2-cat-config-f_W_fix.npz",
        "WaveGAN_piano_MSE": "WaveGAN/spectra_col_WaveGAN_piano_MSE.npz",
        "WaveGAN_piano_STFT": "WaveGAN/spectra_col_WaveGAN_piano_STFT.npz",
        }


corrmat_npz_dict = {"fc6GAN": "fc6GAN/evol_hess_corr_mat.npz",
     "DCGAN": "DCGAN/Hess__corr_mat.npz",
     "BigGAN": "BigGAN/Hess_all_consistency_corr_mat.npz",
    "BigGAN_noise": "BigGAN/Hess_noise_consistency_corr_mat.npz",
    "BigGAN_class": "BigGAN/Hess_class_consistency_corr_mat.npz",
     "BigBiGAN": "BigBiGAN/evol_hess_corr_mat.npz",
     "PGGAN": "PGGAN/Hess__corr_mat.npz",
     "StyleGAN-Face*": "StyleGAN/Hess__corr_mat.npz",
     "StyleGAN2-Face512*": "StyleGAN2/Hess_ffhq-512-avg-tpurun1_corr_mat.npz",
     "StyleGAN2-Face256*": "StyleGAN2/Hess_ffhq-256-config-e-003810_corr_mat.npz",
     "StyleGAN2-Cat256*": "StyleGAN2/Hess_stylegan2-cat-config-f_corr_mat.npz", 
      "StyleGAN-Face_Z": "StyleGAN_Fix/StyleGAN_Face256_fix/Hess_StyleGAN_Face256_fix_corr_mat.npz",
      "StyleGAN2-Face512_Z": "StyleGAN2_Fix/ffhq-512-avg-tpurun1_fix/Hess_ffhq-512-avg-tpurun1_fix_corr_mat.npz",
      "StyleGAN2-Face256_Z": "StyleGAN2_Fix/ffhq-256-config-e-003810_fix/Hess_ffhq-256-config-e-003810_fix_corr_mat.npz",
      "StyleGAN2-Cat256_Z": "StyleGAN2_Fix/stylegan2-cat-config-f_fix/Hess_stylegan2-cat-config-f_fix_corr_mat.npz",
      "StyleGAN2-Face1024_Z": "StyleGAN2_Fix/stylegan2-ffhq-config-f_fix/Hess_StyleGAN_FFHQ1024_corr_mat.npz",
      "StyleGAN-Face_W": "StyleGAN_Fix/StyleGAN_Face256_W_fix/Hess_StyleGAN_Face256_W_fix_corr_mat.npz",
      "StyleGAN2-Face512_W": "StyleGAN2_Fix/ffhq-512-avg-tpurun1_W_fix/Hess_ffhq-512-avg-tpurun1_W_fix_corr_mat.npz",
      "StyleGAN2-Face256_W": "StyleGAN2_Fix/ffhq-256-config-e-003810_W_fix/Hess_ffhq-256-config-e-003810_W_fix_corr_mat.npz",
      "StyleGAN2-Cat256_W": "StyleGAN2_Fix/stylegan2-cat-config-f_W_fix/Hess_stylegan2-cat-config-f_W_fix_corr_mat.npz",
      "WaveGAN_piano_MSE":  "WaveGAN/Hess_WaveGAN_piano_MSE_corr_mat.npz",
      "WaveGAN_piano_STFT": "WaveGAN/Hess_WaveGAN_piano_STFT_corr_mat.npz",
                    }

ctrl_corrmat_npz_dict = {"fc6GAN": "HessNetArchit/FC6GAN/Hess_FC6GAN_shuffle_evol_corr_mat.npz",
     "DCGAN": "HessNetArchit/DCGAN/Hess_DCGAN_shuffle_corr_mat.npz",
     "BigGAN": "HessNetArchit/BigGAN/Hess_BigGAN_shuffle_corr_mat.npz",
    # "BigGAN_noise": "HessNetArchit/BigGAN/Hess_noise_consistency_corr_mat.npz",
    # "BigGAN_class": "HessNetArchit/BigGAN/Hess_class_consistency_corr_mat.npz",
     "BigBiGAN": None, #"HessNetArchit/BigBiGAN/Hess_BigBiGAN_shuffle_corr_mat.npz",
     "PGGAN": "HessNetArchit/PGGAN/Hess_PGGAN_shuffle_corr_mat.npz",
     "StyleGAN-Face*": "HessNetArchit/StyleGAN/Hess_StyleGAN_shuffle_corr_mat.npz",
     "StyleGAN2-Face512*": "HessNetArchit/StyleGAN2/Hess_StyleGAN2_Face512_shuffle_corr_mat.npz",
     # "StyleGAN2-Face256*": "StyleGAN2/Hess_ffhq-256-config-e-003810_corr_mat.npz",
     # "StyleGAN2-Cat256*": "StyleGAN2/Hess_stylegan2-cat-config-f_corr_mat.npz", 
      "StyleGAN-Face_Z": "Hessian_summary/StyleGAN_Fix/StyleGAN_Face256_fix_ctrl/Hess_StyleGAN_Face256_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Face512_Z": "Hessian_summary/StyleGAN2_Fix/ffhq-512-avg-tpurun1_fix_ctrl/Hess_ffhq-512-avg-tpurun1_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Face256_Z": "Hessian_summary/StyleGAN2_Fix/ffhq-256-config-e-003810_fix_ctrl/Hess_ffhq-256-config-e-003810_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Cat256_Z": "Hessian_summary/StyleGAN2_Fix/stylegan2-cat-config-f_fix_ctrl/Hess_stylegan2-cat-config-f_fix_ctrl_corr_mat.npz",
      "StyleGAN-Face_W": "Hessian_summary/StyleGAN_Fix/StyleGAN_Face256_W_fix_ctrl/Hess_StyleGAN_Face256_W_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Face512_W": "Hessian_summary/StyleGAN2_Fix/ffhq-512-avg-tpurun1_W_fix_ctrl/Hess_ffhq-512-avg-tpurun1_W_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Face256_W": "Hessian_summary/StyleGAN2_Fix/ffhq-256-config-e-003810_W_fix_ctrl/Hess_ffhq-256-config-e-003810_W_fix_ctrl_corr_mat.npz",
      "StyleGAN2-Cat256_W": "Hessian_summary/StyleGAN2_Fix/stylegan2-cat-config-f_W_fix_ctrl/Hess_stylegan2-cat-config-f_W_fix_ctrl_corr_mat"
                            ".npz",
                    }

ctrl_Havg_npz_dict = {"fc6GAN": "HessNetArchit/fc6GAN/H_avg_FC6GAN_shuffle_evol.npz",
     "DCGAN": "HessNetArchit/DCGAN/H_avg_DCGAN_shuffle.npz", 
     "BigGAN": "HessNetArchit/BigGAN/H_avg_BigGAN_shuffle.npz",
    # "BigGAN_noise": "BigGAN/H_avg_1000cls.npz",
    # "BigGAN_class": "BigGAN/H_avg_1000cls.npz",
     "BigBiGAN": "HessNetArchit/BigBiGAN/H_avg_BigBiGAN_shuffle.npz",
     "PGGAN": "HessNetArchit/PGGAN/H_avg_PGGAN_shuffle.npz",
     "StyleGAN-Face*": "HessNetArchit/StyleGAN/H_avg_StyleGAN_shuffle.npz",
     # "HessNetArchit/StyleGAN/H_avg_StyleGAN_wspace_shuffle.npz"
     "StyleGAN2-Face512*": "HessNetArchit/StyleGAN2/H_avg_StyleGAN2_Face512_shuffle.npz",
     # "StyleGAN2-Face256*": "StyleGAN2/H_avg_ffhq-256-config-e-003810.npz",
     # "StyleGAN2-Cat256*": "StyleGAN2/H_avg_stylegan2-cat-config-f.npz", 
      "StyleGAN-Face_Z": "Hessian_summary/StyleGAN_Fix/StyleGAN_Face256_fix_ctrl/H_avg_StyleGAN_Face256_fix_ctrl.npz",
      "StyleGAN2-Face512_Z": "Hessian_summary/StyleGAN2_Fix/ffhq-512-avg-tpurun1_fix_ctrl/H_avg_ffhq-512-avg-tpurun1_fix_ctrl.npz",
      "StyleGAN2-Face256_Z": "Hessian_summary/StyleGAN2_Fix/ffhq-256-config-e-003810_fix_ctrl/H_avg_ffhq-256-config-e-003810_fix_ctrl.npz",
      "StyleGAN2-Cat256_Z": "Hessian_summary/StyleGAN2_Fix/stylegan2-cat-config-f_fix_ctrl/H_avg_stylegan2-cat-config-f_fix_ctrl.npz",
      "StyleGAN-Face_W": "Hessian_summary/StyleGAN_Fix/StyleGAN_Face256_W_fix_ctrl/H_avg_StyleGAN_Face256_W_fix_ctrl.npz",
      "StyleGAN2-Face512_W": "Hessian_summary/StyleGAN2_Fix/ffhq-512-avg-tpurun1_W_fix_ctrl/H_avg_ffhq-512-avg-tpurun1_W_fix_ctrl.npz",
      "StyleGAN2-Face256_W": "Hessian_summary/StyleGAN2_Fix/ffhq-256-config-e-003810_W_fix_ctrl/H_avg_ffhq-256-config-e-003810_W_fix_ctrl.npz",
      "StyleGAN2-Cat256_W": "Hessian_summary/StyleGAN2_Fix/stylegan2-cat-config-f_W_fix_ctrl/H_avg_stylegan2-cat-config-f_W_fix_ctrl"
                            ".npz",
                    }

ctrl_spectra_npz_dict = {"fc6GAN": "HessNetArchit/FC6GAN/spectra_col_FC6GAN_shuffle_evol.npz",
          "DCGAN": "HessNetArchit/DCGAN/spectra_col_DCGAN_shuffle.npz",
          "BigGAN": "HessNetArchit/BigGAN/spectra_col_BigGAN_shuffle.npz",
          # "BigGAN_noise": "HessNetArchit/BigGAN/spectra_col_BigGAN_shuffle.npz",
          # "BigGAN_class": "HessNetArchit/BigGAN/spectra_col_BigGAN_shuffle.npz",
          "BigBiGAN": "HessNetArchit/BigBiGAN/spectra_col.npz",
          "PGGAN": "HessNetArchit/PGGAN/spectra_col_PGGAN_shuffle.npz",
          "StyleGAN-Face*": "HessNetArchit/StyleGAN/spectra_col_StyleGAN_shuffle.npz",
          "StyleGAN2-Face512*": "HessNetArchit/StyleGAN2/spectra_col_StyleGAN2_Face512_shuffle.npz",
          # "StyleGAN2-Face256*": "StyleGAN2/spectra_col_ffhq-256-config-e-003810_BP.npz",
          # "StyleGAN2-Cat256*": "StyleGAN2/spectra_col_stylegan2-cat-config-f_BP.npz", 
          "StyleGAN-Face_Z": "Hessian_summary/StyleGAN_Fix/StyleGAN_Face256_fix_ctrl/spectra_col_StyleGAN_Face256_fix_ctrl.npz", 
          "StyleGAN2-Face512_Z": "Hessian_summary/StyleGAN2_Fix/ffhq-512-avg-tpurun1_fix_ctrl/spectra_col_ffhq-512-avg-tpurun1_fix_ctrl.npz", 
          "StyleGAN2-Face256_Z": "Hessian_summary/StyleGAN2_Fix/ffhq-256-config-e-003810_fix_ctrl/spectra_col_ffhq-256-config-e-003810_fix_ctrl.npz", 
          "StyleGAN2-Cat256_Z": "Hessian_summary/StyleGAN2_Fix/stylegan2-cat-config-f_fix_ctrl/spectra_col_stylegan2-cat-config-f_fix_ctrl.npz", 
          "StyleGAN-Face_W": "Hessian_summary/StyleGAN_Fix/StyleGAN_Face256_W_fix_ctrl/spectra_col_StyleGAN_Face256_W_fix_ctrl.npz", 
          "StyleGAN2-Face512_W": "Hessian_summary/StyleGAN2_Fix/ffhq-512-avg-tpurun1_W_fix_ctrl/spectra_col_ffhq-512-avg-tpurun1_W_fix_ctrl.npz", 
          "StyleGAN2-Face256_W": "Hessian_summary/StyleGAN2_Fix/ffhq-256-config-e-003810_W_fix_ctrl/spectra_col_ffhq-256-config-e-003810_W_fix_ctrl.npz", 
          "StyleGAN2-Cat256_W": "Hessian_summary/StyleGAN2_Fix/stylegan2-cat-config-f_W_fix_ctrl/spectra_col_stylegan2-cat-config-f_W_fix_ctrl.npz",  
          }