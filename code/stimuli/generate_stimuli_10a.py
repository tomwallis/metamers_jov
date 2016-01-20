# coding: utf-8

"""First step of stimulus generation for experiment 10.

To understand what this script is doing, see expt_9_stimulus_dev.ipynb.

Tom Wallis wrote it.

"""

##### Import functions that actually do the work ####

import os
import yaml
import pandas as pd
import helpers
import numpy as np
import psyutils as pu
from skimage import io, color
from PIL import Image
from numpy.random import RandomState
from numpy.testing import assert_allclose


"""Params for stimulus generation """

# path to image database:
source_path = '/Users/tomwallis/Dropbox/Image_Databases/mit_1003'

with open('generation_params_exp_10.yaml', 'r') as f:
    params = yaml.load(f)

# params is a dictionary containing the parameter values for stimulus gen.

# set up directories:
top_dir = helpers.project_directory()

source_dir = os.path.join(top_dir, 'stimuli', 'experiment-10',
                          'source_patches')
if not os.path.exists(source_dir):
    os.makedirs(source_dir)

# make an empty master dataframe to be appended with all image codes:
master_dat = pd.DataFrame()


# functions
def out_size(img_name):
    img = Image.open(img_name)
    if np.any(np.asarray(img.size) < params['background_size']):
        out_size = True
    else:
        out_size = False
    return out_size


# random index in case there's a systematic ordering in the database:
def shuffle_order(a, b):
    """ from http://stackoverflow.com/questions/
    4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    """
    rng_state = rng.get_state()
    rng.shuffle(a)
    rng.set_state(rng_state)
    rng.shuffle(b)


def process_bg_im(img_name):
    """ Wrapper to process the background image

    """
    img = io.imread(img_name)
    img = color.rgb2gray(img)
    # crops to image centre:
    img = pu.image.cutout_patch(img, params['background_size'])
    img = check_vals_loop(img, params['rms'])
    return img


def process_fix_im(img_name):
    """ Wrapper to process the fixation image

    """
    img = io.imread(img_name)
    img = color.rgb2gray(img)
    img = pu.image.cutout_patch(img, params['background_size'])
    return img


def renorm_im(im, rms):
    """ Function to renormalise the contrast and mean
    of the image. Assumes image is a float and mean should
    be 0.5 """
    im = pu.image.contrast_image(im, sd=rms/2., returns="contrast")
    im += 0.5

    im[im < 0] = 0
    im[im > 1] = 1
    return im


def check_vals(im, rms):
    # absolute tolerance set so that values round to
    # desired mean / sd...
    assert_allclose(im.mean(), 0.5, atol=0.005)
    assert_allclose(im.std() / im.mean(),
                    rms, atol=0.005)


def check_vals_loop(im, rms):
    im = renorm_im(im, rms)

    try:
        check_vals(im, rms)
    except AssertionError:
        close = False
        while close is False:
            im = renorm_im(im, rms)
            try:
                check_vals(im, rms)
                close = True
            except AssertionError:
                close = False
    return im

"""
Here we source appropriately-sized images from the MIT database

"""
print('Sourcing images...')

# how many of the MIT images would be this large (to allow a 768 sq area)?
img_files, fix_files = helpers.list_mit_ims(source_path)

# list comprehensions to reject outsize images:
img_files = [i for i in img_files if not out_size(i)]
fix_files = [i for i in fix_files if not out_size(i)]

print('Found ' + str(len(img_files)) + ' images of at least 512 px')

# shuffle the order of the image files to prevent any systematic assignment
# of images to sizes:

# set np random state to make results reproducible:
rng = RandomState(2295029)
shuffle_order(img_files, fix_files)

# only using some images:
img_files = img_files[:params['n_ims']]
fix_files = fix_files[:params['n_ims']]

print('Doing image cropping...')

for i, (img_name, fix_name) in enumerate(zip(img_files, fix_files)):
    print('Doing image ' + str(i))
    img = process_bg_im(img_name)
    # img is a 512 crop set to the desired rms.
    fix = process_fix_im(fix_name)

    rms = img.std() / img.mean()

    im_code = os.path.split(img_name)[1][:-5]
    # -5 cuts off file extension.

    out_name = im_code + '.png'
    io.imsave(os.path.join(source_dir, out_name), img)

    # use this image at this size:
    this_dict = {'filename': im_code,
                 'sal_mean': fix.mean(),
                 'sal_sd': fix.std()}

    master_dat = master_dat.append(this_dict,
                                   ignore_index=True)

master_dat.to_csv(os.path.join(source_dir, 'patch_info.csv'), index=False)

pu.files.session_info()
