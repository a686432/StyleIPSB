"""Utility to build montages of images for numpy images array. Similar to that of make_grid"""
import numpy as np
import matplotlib.pylab as plt
from skimage.transform import resize

def decompose_montages(montage, image_shape, montage_shape, padH=0, padW=0, transpose=True, array=True):
    img_list = []
    canvsH, canvsW, _ = montage.shape
    imH, imW = image_shape
    rown, coln = montage_shape
    padH = (canvsH - imH * rown) // (rown + 1)
    padW = (canvsW - imW * coln) // (coln + 1)
    rowfirst = transpose
    for idx in range(rown*coln):
        if rowfirst:
            rn, cn = divmod(idx, coln)
        else:
            cn, rn = divmod(idx, coln) # ??? the parametrization
        img = montage[padH+rn*(imH+padH):(rn+1)*(imH+padH), padW+cn*(imW+padW):(cn+1)*(imW+padW), :]
        img_list.append(img)
    if array:
        img_arr = np.stack(tuple(img_list), axis=3)
        return img_arr
    else:
        return img_list

def build_montages(image_list, image_shape, montage_shape, transpose=True, padH=0, padW=0):
    """Re written
    ---------------------------------------------------------------------------------------------
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format
    ---------------------------------------------------------------------------------------------
    example usage:
    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = make_montages_of_images(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)
    ----------------------------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    
    rsz_list = []
    for img in image_list:
        # if type(img).__module__ != np.__name__:
        #     raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        img = resize(img, image_shape)
        if img.dtype in (np.uint8, np.int) and img.max() > 1.5:  # float 0,1 image
            img = (img / 255.0).astype(np.float64)
        rsz_list.append(img)

    montages = []
    rowfirst = transpose
    # start with black canvas to draw images onto
    imH, imW = image_shape
    rown, coln = montage_shape
    if rowfirst:
        canvas_shape = ((imH + padH) * rown + padH, (imW + padW) * coln + padW, 3)
    else:
        canvas_shape = ((imH + padH) * coln + padH, (imW + padW) * rown + padW, 3)
    montage_image = np.zeros(shape=canvas_shape, dtype=np.float64)

    for idx, img in enumerate(rsz_list):
        pagen, pagei = divmod(idx, (rown*coln))
        if rowfirst:
            cn, rn = divmod(pagei, rown)
        else:
            rn, cn = divmod(pagei, rown) # ??? the parametrization
        montage_image[padH+rn*(imH+padH):(rn+1)*(imH+padH), padW+cn*(imW+padW):(cn+1)*(imW+padW), :] = img
        if (idx + 1) % (rown*coln) == 0:
            montages.append(montage_image)
            montage_image = np.zeros(shape=canvas_shape, dtype=np.float64)# reset black canvas
                
    if (idx + 1) % (rown*coln) != 0:
        montages.append(montage_image)  # add unfinished montage
    return montages

#%% Inspired from MakeGrid in torchvision.utils
def make_grid_np(img_arr, nrow=8, padding=2, pad_value=0):
    nmaps = img_arr.shape[3]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(img_arr.shape[0] + padding), int(img_arr.shape[1] + padding)
    grid = np.zeros((height * ymaps + padding, width * xmaps + padding, 3), dtype=img_arr.dtype)
    grid.fill(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding: (y + 1) * height, x * width + padding: (x + 1) * width, :] = img_arr[:,:,:,k]
            k = k + 1
    return grid


def color_frame(img, color, pad=10):
    outimg = np.ones((img.shape[0] + pad * 2, img.shape[1] + pad * 2, 3))
    outimg = outimg * color[:3]
    outimg[pad:-pad, pad:-pad, :] = img
    return outimg


def color_framed_montages(image_list, image_shape, montage_shape, scores, cmap=plt.cm.summer, pad=24):
    # get color for each cell
    if (not scores is None) and (not cmap is None):
        lb = np.min(scores)
        ub = max(np.max(scores), lb + 0.001)
        colorlist = [cmap((score - lb) / (ub - lb)) for score in scores]
        # pad color to the image
        frame_image_list = [color_frame(img, color, pad=pad) for img, color in zip(image_list, colorlist)]
    else:
        frame_image_list = image_list
    image_montages = []
    # start with black canvas to draw images onto
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                             dtype=np.float64)
    cursor_pos = [0, 0]
    start_new_img = False
    for img in frame_image_list:
        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = resize(img, image_shape)
        if img.dtype in (np.uint8, np.int) and img.max() > 1.0:  # float 0,1 image
            img = (img / 255.0).astype(np.float64)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(
                    shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3), dtype=np.float64)
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages