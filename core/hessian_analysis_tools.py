#%%
"""
This lib curates functions that are useful for Hessian analysis for different GANs
- load computed hess npz
- Visualize spectra
- Visualize consistency of Hessian matrices across position in the latent space
- General pipeline of analysis
"""
import sys
import re
import os
from os.path import join
from glob import glob
from time import time
from tqdm import tqdm
from easydict import EasyDict
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def scan_hess_npz(Hdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP', evckey='evc_BP', featkey=None):
    """ Function to load in npz and collect the spectra.
    Set evckey=None to avoid loading eigenvectors.

    Note for newer experiments use evakey='eva_BP', evckey='evc_BP'
    For older experiments use evakey='eigvals', evckey='eigvects'"""
    npzpaths = glob(join(Hdir, "*.npz"))
    npzfns = [path.split("\\")[-1] for path in npzpaths]
    npzpattern = re.compile(npzpat)
    eigval_col = []
    eigvec_col = []
    feat_col = []
    meta = []
    for fn, path in tqdm(zip(npzfns, npzpaths)):
        match = npzpattern.findall(fn)
        if len(match) == 0:
            continue
        parts = match[0]  # trunc, RND
        data = np.load(path)
        try:
            evas = data[evakey]
            eigval_col.append(evas)
            if evckey is not None:
                evcs = data[evckey]
                eigvec_col.append(evcs)
            if featkey is not None:
                feat = data[featkey]
                feat_col.append(feat)
            meta.append(parts)
        except KeyError:
            print("KeyError, keys in the archive : ", list(data))
            return
    eigval_col = np.array(eigval_col)
    print("Load %d npz files of Hessian info" % len(meta))
    if featkey is None:
        return eigval_col, eigvec_col, meta
    else:
        feat_col = np.array(tuple(feat_col)).squeeze()
        return eigval_col, eigvec_col, feat_col, meta
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
#%%
def plot_spectra(eigval_col, savename="spectrum_all", figdir="", abs=True, median=False,
                 titstr="GAN", label="all", fig=None, save=True):
    """A local function to compute these figures for different subspaces. """
    if abs:
        eigval_col = np.abs(eigval_col)
    if median: eigmean = np.nanmedian(eigval_col, axis=0)
    else: eigmean = np.nanmean(eigval_col, axis=0)
    eiglim = np.percentile(eigval_col, [5, 95], axis=0)
    sortIdx = np.argsort(-np.abs(eigmean))
    eigmean = eigmean[sortIdx]
    eiglim = eiglim[:, sortIdx]
    eigN = len(eigmean)
    if fig is None:
        fig, axs = plt.subplots(1, 2, figsize=[10, 5])
    else:
        plt.figure(num=fig.number)
        axs = fig.axes
    plt.sca(axs[0])
    plt.plot(range(eigN), eigmean, alpha=0.6)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(eigN), eiglim[0, :], eiglim[1, :], alpha=0.3, label="all space")
    plt.ylabel("eigenvalue")
    plt.xlabel("eig id")
    plt.legend()
    plt.sca(axs[1])
    plt.plot(range(eigN), np.log10(eigmean), alpha=0.6)  #, eigval_arr.std(axis=0)
    plt.fill_between(range(eigN), np.log10(eiglim[0, :]), np.log10(eiglim[1, :]), alpha=0.3, label=label)
    plt.ylabel("eigenvalue(log)")
    plt.xlabel("eig id")
    plt.legend()
    st = plt.suptitle("Hessian Spectrum of %s\n (error bar for [5,95] percentile %s)"%(titstr,
                      "median as curve" if median else "mean as curve"))
    if save:
        plt.savefig(join(figdir, savename+".png"), bbox_extra_artists=[st]) # this is working.
        plt.savefig(join(figdir, savename+".pdf"), bbox_extra_artists=[st])  # this is working.
    # plt.show()
    return fig
#%%
import numpy.ma as ma
import torch
def corr_torch(V1, V2):
    C1 = (V1 - V1.mean())
    C2 = (V2 - V2.mean())
    return torch.dot(C1, C2) / C1.norm() / C2.norm()

def corr_nan_torch(V1, V2):
    Msk = torch.isnan(V1) | torch.isnan(V2)
    return corr_torch(V1[~Msk], V2[~Msk])

def compute_hess_corr(eigval_col, eigvec_col, savelabel="", figdir="", use_cuda=False):
    """
    User Note: cuda should be used for large mat mul like 512 1024 4096.
    small matmul should stay with cpu numpy computation. cuda will add the IO overhead.
    """
    posN = len(eigval_col)
    T0 = time()
    if use_cuda:
        corr_mat_log = torch.zeros((posN, posN)).cuda()
        corr_mat_lin = torch.zeros((posN, posN)).cuda()
        for eigi in tqdm(range(posN)):
            evc_i, eva_i = torch.from_numpy(eigvec_col[eigi]).cuda(), torch.from_numpy(eigval_col[eigi]).cuda()
            for eigj in range(posN):
                evc_j, eva_j = torch.from_numpy(eigvec_col[eigj]).cuda(), torch.from_numpy(eigval_col[eigj]).cuda()
                inpr = evc_i.T @ evc_j
                vHv_ij = torch.diag((inpr * eva_j.unsqueeze(0)) @ inpr.T)
                corr_mat_log[eigi, eigj] = corr_nan_torch(vHv_ij.log10(), eva_j.log10())
                corr_mat_lin[eigi, eigj] = corr_nan_torch(vHv_ij, eva_j)
        corr_mat_log = corr_mat_log.cpu().numpy()
        corr_mat_lin = corr_mat_lin.cpu().numpy()
    else:
        corr_mat_log = np.zeros((posN, posN))
        corr_mat_lin = np.zeros((posN, posN))
        for eigi in tqdm(range(posN)):
            eva_i, evc_i = eigval_col[eigi], eigvec_col[eigi]
            for eigj in range(posN):
                eva_j, evc_j = eigval_col[eigj], eigvec_col[eigj]
                inpr = evc_i.T @ evc_j
                vHv_ij = np.diag((inpr * eva_j[np.newaxis, :]) @ inpr.T)
                corr_mat_log[eigi, eigj] = ma.corrcoef(ma.masked_invalid(np.log10(vHv_ij)), ma.masked_invalid(np.log10(eva_j)))[0, 1]
                corr_mat_lin[eigi, eigj] = np.corrcoef(vHv_ij, eva_j)[0, 1]

    print("%.1f sec" % (time() - T0)) # 582.2 secs for the 1000 by 1000 mat. not bad!
    np.savez(join(figdir, "Hess_%s_corr_mat.npz" % savelabel), corr_mat_log=corr_mat_log, corr_mat_lin=corr_mat_lin)
    print("Compute results saved to %s" % join(figdir, "Hess_%s_corr_mat.npz" % savelabel))
    corr_mat_log_nodiag = corr_mat_log.copy()
    corr_mat_lin_nodiag = corr_mat_lin.copy()
    np.fill_diagonal(corr_mat_log_nodiag, np.nan)  # corr_mat_log_nodiag =
    np.fill_diagonal(corr_mat_lin_nodiag, np.nan)  # corr_mat_log_nodiag =
    log_nodiag_mean_cc = np.nanmean(corr_mat_log_nodiag)
    lin_nodiag_mean_cc = np.nanmean(corr_mat_lin_nodiag)
    log_nodiag_med_cc = np.nanmedian(corr_mat_log_nodiag)
    lin_nodiag_med_cc = np.nanmedian(corr_mat_lin_nodiag)
    print("Log scale non-diag mean corr value %.3f med %.3f" % (log_nodiag_mean_cc, log_nodiag_med_cc))
    print("Lin scale non-diag mean corr value %.3f med %.3f" % (lin_nodiag_mean_cc, lin_nodiag_med_cc))
    return corr_mat_log, corr_mat_lin

def compute_vector_hess_corr(eigval_col, eigvec_col, savelabel="", figdir="", use_cuda=False):
    posN = len(eigval_col)
    T0 = time()
    if use_cuda:
        corr_mat_vec = torch.zeros((posN, posN)).cuda()
        for eigi in tqdm(range(posN)):
            eva_i, evc_i = torch.from_numpy(eigval_col[eigi]).cuda(), torch.from_numpy(eigvec_col[eigi]).cuda()
            H_i = (evc_i * eva_i.unsqueeze(0)) @ evc_i.T
            for eigj in range(posN):
                eva_j, evc_j = torch.from_numpy(eigval_col[eigj]).cuda(), torch.from_numpy(eigvec_col[eigj]).cuda()
                H_j = (evc_j * eva_j.unsqueeze(0)) @ evc_j.T
                corr_mat_vec[eigi, eigj] = corr_torch(H_i.flatten(), H_j.flatten())
        corr_mat_vec = corr_mat_vec.cpu().numpy()
    else:
        corr_mat_vec = np.zeros((posN, posN))
        for eigi in tqdm(range(posN)):
            eva_i, evc_i = eigval_col[eigi], eigvec_col[eigi]
            H_i = (evc_i * eva_i[np.newaxis, :]) @ evc_i.T
            for eigj in range(posN):
                eva_j, evc_j = eigval_col[eigj], eigvec_col[eigj]
                H_j = (evc_j * eva_j[np.newaxis, :]) @ evc_j.T
                # corr_mat_log[eigi, eigj] = \
                # np.corrcoef(ma.masked_invalid(np.log10(vHv_ij)), ma.masked_invalid(np.log10(eva_j)))[0, 1]
                corr_mat_vec[eigi, eigj] = np.corrcoef(H_i.flatten(), H_j.flatten())[0, 1]
    print("%.1f sec" % (time() - T0))  #
    np.savez(join(figdir, "Hess_%s_corr_mat_vec.npz" % savelabel), corr_mat_vec=corr_mat_vec, )
    print("Compute results saved to %s" % join(figdir, "Hess_%s_corr_mat_vec.npz" % savelabel))
    return corr_mat_vec
#%
def plot_consistentcy_mat(corr_mat_log, corr_mat_lin, savelabel="", figdir="", titstr="GAN"):
    posN = corr_mat_log.shape[0]
    corr_mat_log_nodiag = corr_mat_log.copy()
    corr_mat_lin_nodiag = corr_mat_lin.copy()
    np.fill_diagonal(corr_mat_log_nodiag, np.nan)
    np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
    log_nodiag_mean_cc = np.nanmean(corr_mat_log_nodiag)
    lin_nodiag_mean_cc = np.nanmean(corr_mat_lin_nodiag)
    log_nodiag_med_cc = np.nanmedian(corr_mat_log_nodiag)
    lin_nodiag_med_cc = np.nanmedian(corr_mat_lin_nodiag)
    print("Log scale non-diag mean corr value %.3f"%np.nanmean(corr_mat_log_nodiag))
    print("Lin scale non-diag mean corr value %.3f"%np.nanmean(corr_mat_lin_nodiag))
    fig1 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_log, fignum=0)
    plt.title("%s Hessian at %d vectors\nCorrelation Mat of log of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, log_nodiag_mean_cc, log_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_corrmat_log.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_corrmat_log.pdf"%savelabel))
    plt.show()

    fig2 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_lin, fignum=0)
    plt.title("%s Hessian at %d vectors\nCorrelation Mat of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, lin_nodiag_mean_cc, lin_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_corrmat_lin.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_corrmat_lin.pdf"%savelabel))
    plt.show()
    return fig1, fig2
#%%
def histogram_corrmat(corr_mat_lin, log=True, GAN="GAN", fig=None, label=""):
    if fig is None:
        fig = plt.figure(figsize=[4, 3])
    else:
        plt.figure(num=fig.number)
    plt.hist(corr_mat_lin.flatten()[~np.isnan(corr_mat_lin.flatten())], 60, density=True, alpha=0.7, label=label)
    corr_mean = np.nanmean(corr_mat_lin)
    corr_medi = np.nanmedian(corr_mat_lin)
    _, YMAX = plt.ylim()
    plt.vlines(corr_mean, 0, YMAX, linestyles="dashed", color="black")
    plt.vlines(corr_medi, 0, YMAX, linestyles="dashed", color="red")
    plt.xlabel("corr(log(V_iH_jV_i), log(Lambda_j))" if log else "corr(V_iH_jV_i, Lambda_j)")
    plt.ylabel("density")
    if fig is not None:
        origtitle = fig.axes[0].get_title()
    else:
        origtitle = ""
    plt.title(origtitle+"Histogram of Non-Diag Correlation\n %s on %s scale\n mean %.3f median %.3f" %
              (GAN, "log" if log else "lin", corr_mean, corr_medi))
    plt.legend()
    # plt.show()
    return fig

def plot_consistency_hist(corr_mat_log, corr_mat_lin, savelabel="", figdir="", titstr="GAN", label="all",
                          figs=(None, None)):
    """Histogram way to represent correlation instead of corr matrix, same interface as plot_consistentcy_mat"""
    posN = corr_mat_log.shape[0]
    np.fill_diagonal(corr_mat_lin, np.nan)
    np.fill_diagonal(corr_mat_log, np.nan)
    if figs is not None: fig1, fig2 = figs
    fig1 = histogram_corrmat(corr_mat_log, log=True, GAN=titstr, fig=fig1, label=label)
    fig1.savefig(join(figdir, "Hess_%s_corr_mat_log_hist.jpg"%savelabel))
    fig1.savefig(join(figdir, "Hess_%s_corr_mat_log_hist.pdf"%savelabel))
    # fig1.show()
    fig2 = histogram_corrmat(corr_mat_lin, log=False, GAN=titstr, fig=fig2, label=label)
    fig2.savefig(join(figdir, "Hess_%s_corr_mat_lin_hist.jpg"%savelabel))
    fig2.savefig(join(figdir, "Hess_%s_corr_mat_lin_hist.pdf"%savelabel))
    # fig2.show()
    return fig1, fig2
#%%
def plot_consistency_example(eigval_col, eigvec_col, nsamp=5, titstr="GAN", figdir="", savelabel=""):
    """
    Note for scatter plot the aspect ratio is set fixed to one.
    :param eigval_col:
    :param eigvec_col:
    :param nsamp:
    :param titstr:
    :param figdir:
    :return:
    """
    Hnums = len(eigval_col)
    eiglist = sorted(np.random.choice(Hnums, nsamp, replace=False))  # range(5)
    fig = plt.figure(figsize=[10, 10], constrained_layout=False)
    spec = fig.add_gridspec(ncols=nsamp, nrows=nsamp, left=0.075, right=0.975, top=0.9, bottom=0.05)
    for axi, eigi in enumerate(eiglist):
        eigval_i, eigvect_i = eigval_col[eigi], eigvec_col[eigi]
        for axj, eigj in enumerate(eiglist):
            eigval_j, eigvect_j = eigval_col[eigj], eigvec_col[eigj]
            inpr = eigvect_i.T @ eigvect_j
            vHv_ij = np.diag((inpr @ np.diag(eigval_j)) @ inpr.T)
            ax = fig.add_subplot(spec[axi, axj])
            if axi == axj:
                ax.hist(np.log10(eigval_j), 20)
            else:
                ax.scatter(np.log10(eigval_j), np.log10(vHv_ij), s=15, alpha=0.6)
                ax.set_aspect(1, adjustable='datalim')
            if axi == nsamp-1:
                ax.set_xlabel("eigvals @ pos %d" % eigj)
            if axj == 0:
                ax.set_ylabel("vHv eigvec @ pos %d" % eigi)
    ST = plt.suptitle("Consistency of %s Hessian Across Vectors\n"
                      "Cross scatter of EigenValues and vHv values for Hessian at %d Random Vectors"%(titstr, nsamp),
                      fontsize=18)
    # plt.subplots_adjust(left=0.175, right=0.95 )
    RND = np.random.randint(1000)
    plt.savefig(join(figdir, "Hess_consistency_example_%s_rnd%03d.jpg" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    plt.savefig(join(figdir, "Hess_consistency_example_%s_rnd%03d.pdf" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    return fig


def hessian_summary_pipeline(savedir, modelnm, figdir, npzpatt="Hess_BP_(\d*).npz", featkey="feat",
                             evakey='eva_BP', evckey='evc_BP', ):
    """ This script crunch down a collection of Hessian computed for a GAN and output proper statistics
    for it. Simple Quick end to end analysis
    savedir: contains an array of npz files with file name pattern npzpatt default `Hess_BP_(\d*).npz`
    featkey, evakey, evckey: Keys to read the refvector, eva, evc in the npz file
    """
    eva_col, evc_col, feat_col, meta = scan_hess_npz(savedir, npzpatt, featkey=featkey, evakey=evakey, evckey=evckey)
    # compute the Mean Hessian and save
    H_avg, eva_avg, evc_avg = average_H(eva_col, evc_col, )
    np.savez(join(figdir, "H_avg_%s.npz" % modelnm), H_avg=H_avg, eva_avg=eva_avg, evc_avg=evc_avg, feats=feat_col)
    # compute and plot spectra
    fig0 = plot_spectra(eigval_col=eva_col, savename="%s_spectrum" % modelnm, figdir=figdir)
    np.savez(join(figdir, "spectra_col_%s.npz" % modelnm), eigval_col=eva_col, )
    # compute and plot the correlation between hessian at different points
    corr_mat_log, corr_mat_lin = compute_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False, savelabel=modelnm)
    corr_mat_vec = compute_vector_hess_corr(eva_col, evc_col, figdir=figdir, use_cuda=False, savelabel=modelnm)
    fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s" % modelnm,
                                       savelabel=modelnm)
    fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=figdir, titstr="%s" % modelnm,
                                         savelabel=modelnm)
    fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=5, titstr="%s" % modelnm, savelabel=modelnm)
    fig3 = plot_consistency_example(eva_col, evc_col, figdir=figdir, nsamp=3, titstr="%s" % modelnm, savelabel=modelnm)
    S = EasyDict({"eva_col": eva_col, "evc_col": evc_col, "feat_col": feat_col, "meta": meta,
                  "H_avg": H_avg, "eva_avg": eva_avg, "evc_avg": evc_avg,
                  "corr_mat_log": corr_mat_log, "corr_mat_lin": corr_mat_lin, "corr_mat_vec": corr_mat_vec,
                  "npzpatt": npzpatt, "featkey": featkey, "evakey": evakey, "evckey": evckey, "modelnm": modelnm,
                  "savedir": savedir})
    return S


#%%
def plot_layer_consistency_mat(corr_mat_log, corr_mat_lin, corr_mat_vec, layernames=None, savelabel="", figdir="", titstr="GAN"):
    """How Hessian matrix in different layers correspond to each other. """
    posN = corr_mat_log.shape[0]
    corr_mat_log_nodiag = corr_mat_log.copy()
    corr_mat_lin_nodiag = corr_mat_lin.copy()
    corr_mat_vec_nodiag = corr_mat_vec.copy()
    np.fill_diagonal(corr_mat_log_nodiag, np.nan)
    np.fill_diagonal(corr_mat_lin_nodiag, np.nan)
    np.fill_diagonal(corr_mat_vec_nodiag, np.nan)
    log_nodiag_mean_cc = np.nanmean(corr_mat_log_nodiag)
    lin_nodiag_mean_cc = np.nanmean(corr_mat_lin_nodiag)
    vec_nodiag_mean_cc = np.nanmean(corr_mat_vec_nodiag)
    log_nodiag_med_cc = np.nanmedian(corr_mat_log_nodiag)
    lin_nodiag_med_cc = np.nanmedian(corr_mat_lin_nodiag)
    vec_nodiag_med_cc = np.nanmedian(corr_mat_vec_nodiag)
    print("Log scale corr non-diag mean value %.3f"%log_nodiag_mean_cc)
    print("Lin scale corr non-diag mean value %.3f"%lin_nodiag_mean_cc)
    print("Vec Hessian corr non-diag mean value %.3f"%vec_nodiag_mean_cc)
    fig1 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_log, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of log of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, log_nodiag_mean_cc, log_nodiag_med_cc), fontsize=15)
    plt.colorbar()
    fig1.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN-0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_log.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_log.pdf"%savelabel))
    plt.show()

    fig2 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_lin, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of vHv and eigenvalues"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, posN, lin_nodiag_mean_cc, lin_nodiag_med_cc), fontsize=15)
    fig2.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN - 0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_lin.jpg"%savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_lin.pdf"%savelabel))
    plt.show()

    fig3 = plt.figure(figsize=[10, 8])
    plt.matshow(corr_mat_vec, fignum=0)
    plt.title("%s Hessian at 1 vector for %d layers\nCorrelation Mat of vectorized Hessian Mat"
              "\nNon-Diagonal mean %.3f median %.3f" % (titstr, posN, vec_nodiag_mean_cc, vec_nodiag_med_cc),
              fontsize=15)
    plt.colorbar()
    fig3.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(posN), layernames);
        plt.ylim(-0.5, posN - 0.5)
        plt.xticks(range(posN), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, posN - 0.5)
    plt.subplots_adjust(top=0.85)
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_vecH.jpg" % savelabel))
    plt.savefig(join(figdir, "Hess_%s_Layer_corrmat_vecH.pdf" % savelabel))
    plt.show()
    return fig1, fig2, fig3

#%%
def plot_layer_consistency_example(eigval_col, eigvec_col, layernames, layeridx=[0,1,-1], titstr="GAN", figdir="", savelabel="", use_cuda=False):
    """
    Note for scatter plot the aspect ratio is set fixed to one.
    :param eigval_col:
    :param eigvec_col:
    :param nsamp:
    :param titstr:
    :param figdir:
    :return:
    """
    nsamp = len(layeridx)
    # Hnums = len(eigval_col)
    # eiglist = sorted(np.random.choice(Hnums, nsamp, replace=False))  # range(5)
    print("Plot hessian of layers : ", [layernames[idx] for idx in layeridx])
    fig = plt.figure(figsize=[10, 10], constrained_layout=False)
    spec = fig.add_gridspec(ncols=nsamp, nrows=nsamp, left=0.075, right=0.975, top=0.9, bottom=0.05)
    for axi, Li in enumerate(layeridx):
        eigval_i, eigvect_i = eigval_col[Li], eigvec_col[Li]
        for axj, Lj in enumerate(layeridx):
            eigval_j, eigvect_j = eigval_col[Lj], eigvec_col[Lj]
            inpr = eigvect_i.T @ eigvect_j
            vHv_ij = np.diag((inpr @ np.diag(eigval_j)) @ inpr.T)
            ax = fig.add_subplot(spec[axi, axj])
            if axi == axj:
                ax.hist(np.log10(eigval_j), 20)
            else:
                ax.scatter(np.log10(eigval_j), np.log10(vHv_ij), s=15, alpha=0.6)
                ax.set_aspect(1, adjustable='datalim')
            if axi == nsamp-1:
                ax.set_xlabel("eigvals @ %s" % layernames[Lj])
            if axj == 0:
                ax.set_ylabel("vHv eigvec @ %s" % layernames[Li])
    ST = plt.suptitle("Consistency of %s Hessian Across Layers\n"
                      "Cross scatter of EigenValues and vHv values for Hessian at %d Layers"%(titstr, nsamp),
                      fontsize=18)
    # plt.subplots_adjust(left=0.175, right=0.95 )
    RND = np.random.randint(1000)
    plt.savefig(join(figdir, "Hess_layer_consistency_example_%s_rnd%03d.jpg" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    plt.savefig(join(figdir, "Hess_layer_consistency_example_%s_rnd%03d.pdf" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    return fig

def plot_layer_mat(layer_mat, layernames=None, titstr="Correlation of Amplification in BigGAN"):
    """Local formatting function for ploting Layer by Layer matrix, used in `compute_plot_layer_corr_mat` """
    Lnum = layer_mat.shape[0]
    fig = plt.figure(figsize=[9, 8])
    plt.matshow(layer_mat, fignum=0)
    layermat_nan = layer_mat.copy()
    np.fill_diagonal(layermat_nan, np.nan)
    plt.title("%s across %d layers"
              "\nNon-Diagonal mean %.3f median %.3f"%(titstr, Lnum, np.nanmean(layermat_nan), np.nanmedian(layermat_nan)), fontsize=15)
    fig.axes[0].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    if layernames is not None:
        plt.yticks(range(Lnum), layernames)
        plt.ylim(-0.5, Lnum - 0.5)
        plt.xticks(range(Lnum), layernames, rotation=35, rotation_mode='anchor', ha='right')
        plt.xlim(-0.5, Lnum - 0.5)
    plt.colorbar()
    plt.subplots_adjust(top=0.85)
    plt.show()
    return fig

def compute_plot_layer_corr_mat(eva_col, evc_col, H_col, layernames, titstr="BigGAN", savestr="BigGAN", figdir="", use_cuda=False):
    Lnum = len(evc_col)
    corr_mat_lin = np.zeros((Lnum, Lnum))
    corr_mat_log = np.zeros((Lnum, Lnum))
    log_reg_slope = np.zeros((Lnum, Lnum))
    log_reg_intcp = np.zeros((Lnum, Lnum))
    for Li in range(Lnum):
        eva, evc = eva_col[Li], evc_col[Li]
        eva_i, evc_i = (eva, evc) if not use_cuda else \
            (torch.from_numpy(eva).cuda(), torch.from_numpy(evc).cuda())
        for Lj in range(Lnum):  # hessian target
            if H_col is not None:
                H = H_col[Lj]
                if use_cuda:
                    H = torch.from_numpy(H).cuda()
                    alphavec = torch.diag(evc_i.T @ H @ evc_i).cpu().numpy()
                else:
                    alphavec = np.diag(evc_i.T @ H @ evc_i)
            else:
                if use_cuda:
                    eva_j, evc_j = torch.from_numpy(eva_col[Lj]).cuda(), torch.from_numpy(evc_col[Lj]).cuda()
                    inpr = evc_i.T @ evc_j
                    alphavec = torch.diag((inpr * eva_j.view(1, -1)) @ inpr.T)
                    alphavec = alphavec.cpu().numpy()
                else:
                    inpr = evc_i.T @ evc_col[Lj]
                    # H = evc_col[Lj] @ np.diag(eva_col[Lj]) @ evc_col[Lj].T
                    alphavec = np.diag((inpr * eva_col[Lj].reshape(1, -1)) @ inpr.T)
    # for Li in range(Lnum):
    #     evc = evc_col[Li]
    #     eva = eva_col[Li]
    #     for Lj in range(Lnum):
    #         H = H_col[Lj]
    #         alphavec = np.diag(evc.T @ H @ evc)
            log10alphavec = np.log10(alphavec)
            log10eva = np.log10(eva)
            corr_mat_lin[Li, Lj] = np.corrcoef(alphavec, eva)[0,1]
            corr_mat_log[Li, Lj] = ma.corrcoef(ma.masked_invalid(log10alphavec), ma.masked_invalid(log10eva))[0, 1]  #np.corrcoef(log10alphavec, log10eva)[0,1]
            nanmask = (~np.isnan(log10alphavec)) * (~np.isnan(log10eva))
            slope, intercept = np.polyfit(log10eva[nanmask], log10alphavec[nanmask], 1)
            log_reg_slope[Li, Lj] = slope
            log_reg_intcp[Li, Lj] = intercept
    fig1 = plot_layer_mat(corr_mat_lin, layernames=layernames, titstr="Linear Correlation of Amplification in %s"%titstr)
    fig1.savefig(join(figdir, "%s_Layer_corr_lin_mat.pdf"%savestr))
    fig2 = plot_layer_mat(corr_mat_log, layernames=layernames, titstr="Log scale Correlation of Amplification in %s"%titstr)
    fig2.savefig(join(figdir, "%s_Layer_corr_log_mat.pdf"%savestr))
    fig3 = plot_layer_mat(log_reg_slope, layernames=layernames, titstr="Log scale Slope of Amplification in %s"%titstr)
    fig3.savefig(join(figdir, "%s_Layer_log_reg_slope.pdf"%savestr))
    fig4 = plot_layer_mat(log_reg_intcp, layernames=layernames, titstr="Log scale intercept of Amplification in %s"%titstr)
    fig4.savefig(join(figdir, "%s_Layer_log_reg_intercept.pdf"%savestr))
    return corr_mat_lin, corr_mat_log, log_reg_slope, log_reg_intcp, fig1, fig2, fig3, fig4,

def plot_layer_amplif_curves(eva_col, evc_col, H_col, layernames, savestr="", figdir="",
                             maxnorm=False, use_cuda=False):
    Lnum = len(evc_col)
    colorseq = [cm.jet(Li / (Lnum - 1)) for Li in range(Lnum)]  # color for each curve.
    for Li in range(Lnum):  # source of eigenvector basis
        alphavec_col = []
        if use_cuda:
            eva_i, evc_i = torch.from_numpy(eva_col[Li]).cuda(), torch.from_numpy(evc_col[Li]).cuda()
        else:
            eva_i, evc_i = eva_col[Li], evc_col[Li]
        plt.figure(figsize=[5, 4])
        for Lj in range(Lnum):  # hessian target
            if H_col is not None:
                H = H_col[Lj]
                if use_cuda:
                    H = torch.from_numpy(H).cuda()
                    alphavec = torch.diag(evc_i.T @ H @ evc_i).cpu().numpy()
                else:
                    alphavec = np.diag(evc_i.T @ H @ evc_i)
            else:
                if use_cuda:
                    eva_j, evc_j = torch.from_numpy(eva_col[Lj]).cuda(), torch.from_numpy(evc_col[Lj]).cuda()
                    inpr = evc_i.T @ evc_j
                    alphavec = torch.diag((inpr * eva_j.view(1, -1)) @ inpr.T)
                    alphavec = alphavec.cpu().numpy()
                else:
                    inpr = evc_i.T @ evc_col[Lj]
                    # H = evc_col[Lj] @ np.diag(eva_col[Lj]) @ evc_col[Lj].T
                    alphavec = np.diag((inpr * eva_col[Lj].reshape(1,-1)) @ inpr.T)
            alphavec_col.append(alphavec)
            scaler = alphavec[-1] if maxnorm else 1
            plt.plot(np.log10(alphavec[::-1] / scaler), label=layernames[Lj], color=colorseq[Lj], lw=2, alpha=0.7)
        plt.xlabel("Rank of eigenvector (layer %d %s)" % (Li, layernames[Li]))
        plt.ylabel("Amplification (normalize max to 1)" if maxnorm else "Amplification")  # (layer %d %s)"%(Lj, layernames[Lj]
        plt.title("Amplification factor of layer %s eigenvector in all layers"%layernames[Li])
        plt.legend()
        plt.savefig(join(figdir, "%s_Ampl_curv_evc_Layer%d%s.png" % (savestr, Li, "_mxnorm" if maxnorm else "")))
        plt.savefig(join(figdir, "%s_Ampl_curv_evc_Layer%d%s.pdf" % (savestr, Li, "_mxnorm" if maxnorm else "")))
        plt.show()

def plot_layer_amplif_consistency(eigval_col, eigvec_col, layernames, layeridx=[0,1,-1], titstr="GAN", figdir="",
                                   savelabel="", src_eigval=True):
    """ Plot the consistency matrix just like the spatial Hessian consistency. Resulting figure will have len(layeridx) by
        len(layeridx) panels.
    layeridx: index of layer to plot, index in layernames and eigval_col, eigvec_col.
    """
    nsamp = len(layeridx)
    print("Plot hessian of layers : ", [layernames[idx] for idx in layeridx])
    fig = plt.figure(figsize=[10, 10], constrained_layout=False)
    spec = fig.add_gridspec(ncols=nsamp, nrows=nsamp, left=0.075, right=0.975, top=0.9, bottom=0.05)
    for axi, Li in enumerate(layeridx):
        eigval_i, eigvect_i = eigval_col[Li], eigvec_col[Li]
        for axj, Lj in enumerate(layeridx):
            eigval_j, eigvect_j = eigval_col[Lj], eigvec_col[Lj]
            inpr = eigvect_i.T @ eigvect_j
            vHv_ij = np.diag((inpr @ np.diag(eigval_j)) @ inpr.T)
            ax = fig.add_subplot(spec[axi, axj])
            if axi == axj:
                ax.hist(np.log10(eigval_i), 20)
            else:
                if src_eigval:  # plot eigen values of source layer on x axis
                    ax.scatter(np.log10(eigval_i), np.log10(vHv_ij), s=15, alpha=0.6)
                else:  # plot eigen values of target layers on x axis
                    ax.scatter(np.log10(eigval_j), np.log10(vHv_ij), s=15, alpha=0.6)
                ax.set_aspect(1, adjustable='datalim')
            if axi == nsamp - 1:
                ax.set_xlabel("H @ %s" % layernames[Lj])
            if axj == 0:
                ax.set_ylabel("eigvect @ %s" % layernames[Li])
    ST = plt.suptitle("Consistency of %s Amplification Factor Across Layers\n"
                      "Scatter of AmpFact for Hessian at %d Layers\n Source of EigVect on y axes, Source of Hessian "
                      "on x axes" % (titstr, nsamp),
                      fontsize=18)
    # plt.subplots_adjust(left=0.175, right=0.95 )
    RND = np.random.randint(1000)
    plt.savefig(join(figdir, "Amplif_layer_consistency_example_%s_rnd%03d.jpg" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    plt.savefig(join(figdir, "Amplif_layer_consistency_example_%s_rnd%03d.pdf" % (savelabel, RND)),
                bbox_extra_artists=[ST])  #
    return fig



#%% Section: Hessian comparison at the same location. Different metrics to compare Hessian.
def spectra_cmp(eigvals1, eigvals2, show=True):
    cc = np.corrcoef((eigvals1), (eigvals2))[0, 1]
    logcc = np.corrcoef(np.log10(np.abs(eigvals1)+1E-8), np.log10(np.abs(eigvals2)+1E-8))[0, 1]
    reg_coef = np.polyfit((eigvals1), (eigvals2), 1)
    logreg_coef = np.polyfit(np.log10(np.abs(eigvals1)+1E-8), np.log10(np.abs(eigvals2)+1E-8), 1)
    if show:
        print("Correlation %.3f (lin) %.3f (log). Regress Coef [%.2f, %.2f] (lin) [%.2f, %.2f] (log)"%
            (cc, logcc, *tuple(reg_coef), *tuple(logreg_coef)))
    return cc, logcc, reg_coef, logreg_coef

def Hessian_cmp(eigvals1, eigvecs1, H1, eigvals2, eigvecs2, H2, show=True):
    H_cc = np.corrcoef(H1.flatten(), H2.flatten())[0,1]
    logH1 = eigvecs1 * np.log10(np.abs(eigvals1))[np.newaxis, :] @ eigvecs1.T
    logH2 = eigvecs2 * np.log10(np.abs(eigvals2))[np.newaxis, :] @ eigvecs2.T
    logH_cc = np.corrcoef(logH1.flatten(), logH2.flatten())[0, 1]
    if show:
        print("Entrywise Correlation Hessian %.3f log Hessian %.3f (log)"% (H_cc, logH_cc,))
    return H_cc, logH_cc

def top_eigvec_corr(eigvects1, eigvects2, eignum=10):
    cc_arr = []
    for eigi in range(eignum):
        cc = np.corrcoef(eigvects1[:, -eigi-1], eigvects2[:, -eigi-1])[0, 1]
        cc_arr.append(cc)
    return np.abs(cc_arr)

def eigvec_H_corr(eigvals1, eigvects1, H1, eigvals2, eigvects2, H2, show=True):
    vHv12 = np.diag(eigvects1.T @ H2 @ eigvects1)
    vHv21 = np.diag(eigvects2.T @ H1 @ eigvects2)
    cc_12 = np.corrcoef(vHv12, eigvals2)[0, 1]
    cclog_12 = np.corrcoef(np.log(np.abs(vHv12)+1E-8), np.log(np.abs(eigvals2+1E-8)))[0, 1]
    cc_21 = np.corrcoef(vHv21, eigvals1)[0, 1]
    cclog_21 = np.corrcoef(np.log(np.abs(vHv21)+1E-8), np.log(np.abs(eigvals1+1E-8)))[0, 1]
    if show:
        print("Applying eigvec 1->2: corr %.3f (lin) %.3f (log)"%(cc_12, cclog_12))
        print("Applying eigvec 2->1: corr %.3f (lin) %.3f (log)"%(cc_21, cclog_21))
    return cc_12, cclog_12, cc_21, cclog_21