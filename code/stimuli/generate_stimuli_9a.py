# coding: utf-8

"""First step of stimulus generation for experiment 9.

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

with open('generation_params_exp_9.yaml', 'r') as f:
    params = yaml.load(f)

# params is a dictionary containing the parameter values for stimulus gen.

# set up directories:
top_dir = helpers.project_directory()

inner_dir = os.path.join(top_dir, 'stimuli', 'experiment-9',
                         'inner_patches')
if not os.path.exists(inner_dir):
    os.makedirs(inner_dir)

middle_dir = os.path.join(top_dir, 'stimuli', 'experiment-9',
                          'middle_patches')
if not os.path.exists(middle_dir):
    os.makedirs(middle_dir)

outer_dir = os.path.join(top_dir, 'stimuli', 'experiment-9',
                         'outer_patches')
if not os.path.exists(outer_dir):
    os.makedirs(outer_dir)

# make an empty master dataframe to be appended with all image codes:
master_dat = pd.DataFrame()


# functions
def out_size(img_name):
    img = Image.open(img_name)
    if np.any(np.asarray(img.size) < 768):
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


def crop_patch(img, size, centre):
    """
    :param img: the image to crop from
    :param size: side length of desired crop
    :param centre: horizontal centre of desired crop.
    :returns: cropped image, rms contrast value
    """

    img = pu.image.cutout_patch(img, size, (centre, mid_y))
    rms = img.std() / img.mean()

    return img, rms


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

# shuffle the order of the image files to prevent any systematic assignment
# of images to sizes:

# set np random state to make results reproducible:
rng = RandomState(1128495)
shuffle_order(img_files, fix_files)

"""
For each image,

- for each potential patch size,
    - check the rms in each patch exceeds the 'patch_rms_min' value
    - if so, assign image to this patch size, crop patches and save
    - if not, go onto the next size
    - if final size exceeded, can't use the image.

"""


print('Doing image cropping for inner, middle and outer patches...')

assigned_counter = np.zeros(len(params['sizes']), dtype=np.int)
nominal_ims_per_size = int(np.floor(len(img_files) / len(params['sizes'])))
mid_y = params['background_size'] / 2

for i, (img_name, fix_name) in enumerate(zip(img_files, fix_files)):
    print('Doing image ' + str(i))
    img = process_bg_im(img_name)
    assigned = False
    j = 0
    # check sizes:
    while not assigned:
        size = params['sizes'][j]
        inner_pos = params['inner_patch_img'][j]
        outer_pos = params['outer_patch_img'][j]

        middle, middle_rms = crop_patch(img, size, params['middle_centre_img'])
        inner, inner_rms = crop_patch(img, params['inner'], inner_pos)
        outer, outer_rms = crop_patch(img, params['outer'], outer_pos)

        if (middle_rms > params['patch_rms_min']) & \
           (inner_rms > params['patch_rms_min']) & \
           (outer_rms > params['patch_rms_min']) & \
           (assigned_counter[j] < nominal_ims_per_size):
            print('assigning image to size ' + str(size))

            fix = process_fix_im(fix_name)

            middle_fix, rms = crop_patch(fix, size, params['middle_centre_img'])
            inner_fix, rms = crop_patch(fix, params['inner'], inner_pos)
            outer_fix, rms = crop_patch(fix, params['outer'], outer_pos)

            im_code = os.path.split(img_name)[1][:-5]
            # -5 cuts off file extension.

            middle_file = im_code + '_middle.png'
            inner_file = im_code + '_inner.png'
            outer_file = im_code + '_outer.png'

            io.imsave(os.path.join(middle_dir, middle_file), middle)
            io.imsave(os.path.join(inner_dir, inner_file), inner)
            io.imsave(os.path.join(outer_dir, outer_file), outer)

            # use this image at this size:
            this_dict = {'filename': im_code,
                         'size': size,
                         'middle_rms': middle_rms,
                         'inner_rms': inner_rms,
                         'outer_rms': outer_rms,
                         'sal_mid_mean': middle_fix.mean(),
                         'sal_mid_sd': middle_fix.std(),
                         'sal_inner_mean': inner_fix.mean(),
                         'sal_inner_sd': inner_fix.std(),
                         'sal_outer_mean': outer_fix.mean(),
                         'sal_outer_sd': outer_fix.std()}

            master_dat = master_dat.append(this_dict,
                                           ignore_index=True)
            # increment counter
            assigned_counter[j] += 1
            # image is assigned:
            assigned = True
        elif j == len(params['sizes'])-1:
            print('Failed to allocate image ' + img_name)
            print(inner_rms, middle_rms, outer_rms)
            assigned = True
        else:
            j += 1

print(assigned_counter)

master_dat.to_csv(os.path.join(middle_dir, 'patch_info.csv'), index=False)

pu.files.session_info()
