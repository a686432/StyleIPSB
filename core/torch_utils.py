import sys
import time
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
# utils to show 3d numpy array.
def show_npimage(mtg, title=""):
    if mtg.dtype is not np.uint8:
        if np.max(mtg) < 1.2:
            Image.fromarray((255 * np.clip(mtg, 0, 1)).astype(np.uint8)).show(title)
        else:
            Image.fromarray((np.clip(mtg, 0, 255)).astype(np.uint8)).show(title)
    else:
        Image.fromarray(mtg).show(title)

# utils to show or save 4d torch tensors as grid of images.
def show_imgrid(img_tsr, *args, **kwargs):
    if type(img_tsr) is list:
        if img_tsr[0].ndim == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif img_tsr[0].ndim == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)
    PILimg = ToPILImage()(make_grid(img_tsr.cpu(), *args, **kwargs))
    PILimg.show()
    return PILimg

def save_imgrid(img_tsr, path, *args, **kwargs):
    PILimg = ToPILImage()(make_grid(img_tsr.cpu(), *args, **kwargs))
    PILimg.save(path)
    return PILimg

# Utils below are fetched from `hessian_eigenthings` 
#       https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/8ff8b3907f2383fe1fdaa232736c8fef295d8131/hessian_eigenthings/utils.py#L19
import shutil
def maybe_fp16(vec, fp16):
    return vec.half() if fp16 else vec.float()

TOTAL_BAR_LENGTH = 65.0
term_width = shutil.get_terminal_size().columns
last_time = time.time()
begin_time = last_time

def format_time(seconds):
    """ converts seconds into day-hour-minute-second-ms string format """
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def progress_bar(current, total, msg=None):
    """ handy utility to display an updating progress bar...
    percentage completed is computed as current/total
    from: https://github.com/noahgolmant/skeletor/blob/master/skeletor/utils.py
    """
    global last_time, begin_time  # pylint: disable=global-statement
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for _ in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for _ in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for _ in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for _ in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()