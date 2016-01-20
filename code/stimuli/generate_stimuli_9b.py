# coding: utf-8

"""Second step of stimulus generation for experiment 9. This script will
take the source patches and synths, window them, crop them, and resize
if appropriate, before saving to final_ims.

You could run it by typing

    python generate_stimuli_9b.py

at the command line, assuming you have all the necessary packages and
the appropriate python environment installed.

This script assumes you've run `generate_stimuli_9a.py`, then

`p_s_generation_expt_9_....m`

To understand what this script is doing, see expt_8_stimulus_dev.ipynb.

Tom Wallis wrote it.

"""

##### Import functions that actually do the work ####

import os
import yaml
import pandas as pd
import helpers
import numpy as np
import psyutils as pu
from skimage import io, color, img_as_float

"""Params for stimulus generation """

with open('generation_params_exp_9.yaml', 'r') as f:
    params = yaml.load(f)

# params is a dictionary containing the parameter values for stimulus gen.

# set up directories:
top_dir = helpers.project_directory()

middle_patches = os.path.join(top_dir, 'stimuli', 'experiment-9',
                              'middle_patches')
middle_synths = os.path.join(top_dir, 'stimuli', 'experiment-9',
                             'middle_synths')
inner_patches = os.path.join(top_dir, 'stimuli', 'experiment-9',
                             'inner_patches')
inner_synths = os.path.join(top_dir, 'stimuli', 'experiment-9',
                            'inner_synths')
outer_patches = os.path.join(top_dir, 'stimuli', 'experiment-9',
                             'outer_patches')
outer_synths = os.path.join(top_dir, 'stimuli', 'experiment-9',
                            'outer_synths')

out_dir = os.path.join(top_dir, 'stimuli', 'experiment-9',
                       'final_ims')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# make an empty master dataframe to be appended with all image codes:
master_dat = pd.DataFrame()


# define main image treatment functions:
def window_im(im):
    # window in cosine:
    win = pu.image.cos_win_2d(size=im.shape[0],
                              ramp=10, ramp_type='pixels')

    # unfortunately need to do windowing by conversion to RGBA.
    # makes all file sizes bigger...
    rgba = np.ndarray((im.shape[0], im.shape[1], 4),
                      dtype=np.float)
    rgba[..., :3] = color.gray2rgb(img_as_float(im))
    rgba[..., 3] = win

    return rgba


def do_patch(in_name,
             out_name,
             desired_size):
    """
    :param in_name: the absolute path to the input (middle) patch
    :param desired_size: the desired size for this patch.
    :param out_name: the absolute path to the output image.
    """

    # load image:
    im = io.imread(in_name)

    # resize if desired:
    if im.shape[0] != desired_size:
        # # resizing ensures that features are same but rescaled:
        # im = transform.resize(im, (dat.ix[idx, 'new_size'],
        #                            dat.ix[idx, 'new_size']))

        # cropping keeps object-pixel relationship
        # same as in other levels. I will use it here for similarity
        # to the texture images:
        im = pu.image.cutout_patch(im, desired_size)

    im = window_im(im)  # returns an rgba float.

    # save:
    io.imsave(out_name, im)

""" For each desired size, resize patch and synths if required.
Alpha blend with background ims, leaving a grey border..."""

dat = pd.read_csv(os.path.join(middle_patches, 'patch_info.csv'))

# want to resize half the 256 and 512 ims:
dat['new_size'] = dat['size']
downsize_levels = [256, 512]
desired_sizes = [192, 384]
dat['downsampled'] = False

for size, desired in zip(downsize_levels, desired_sizes):
    # select half of the images with this_size:
    mask = dat['size'] == size
    sub_dat = dat.loc[mask, :]
    # get index of first half of these values:
    idx = sub_dat.index[0:len(sub_dat.index)//2]  # select first half of images
    # modify larger data frame:
    dat.ix[idx, 'new_size'] = desired
    dat.ix[idx, 'downsampled'] = True


# surround conditions:
surr_source = ['nat', 'synth']
fail_count = 0
# now we know the size and desired size, loop over ims,
# resizing if necessary, then window and save:
for idx in range(len(dat.index)):
    print('Scene ' + str(idx))
    im_code = dat.ix[idx, 'filename']

    # check that all files exist, or skip this image (in case patch can't be
    # synthed in some images):
    inner_synth = os.path.join(inner_synths,
                               (im_code + '_inner_synth_1.png'))
    outer_synth = os.path.join(outer_synths,
                               (im_code + '_outer_synth_1.png'))
    middle_synth_1 = os.path.join(middle_synths,
                                  (im_code + '_middle_synth_1.png'))
    middle_synth_2 = os.path.join(middle_synths,
                                  (im_code + '_middle_synth_2.png'))
    middle_synth_3 = os.path.join(middle_synths,
                                  (im_code + '_middle_synth_3.png'))

    if os.path.exists(inner_synth) and \
            os.path.exists(outer_synth) and \
            os.path.exists(middle_synth_1) and \
            os.path.exists(middle_synth_2) and \
            os.path.exists(middle_synth_3):

        # natural target patch:
        in_path = os.path.join(middle_patches, (im_code + '_middle.png'))
        out_path = os.path.join(out_dir,
                                (im_code + '_mid_natural.png'))
        desired_size = dat.ix[idx, 'new_size']
        do_patch(in_path, out_path, desired_size)

        # synthesized target patches:
        for synth in range(3):
            synth += 1
            in_path = os.path.join(middle_synths,
                                   (im_code + '_middle_' +
                                    'synth_' + str(synth) + '.png'))

            out_path = os.path.join(out_dir,
                                    (im_code +
                                     '_mid_synth_' + str(synth) + '.png'))
            do_patch(in_path, out_path, desired_size)

        # surround patches:
        for source in surr_source:
            if source is 'nat':
                inner_path = os.path.join(inner_patches,
                                          (im_code + '_inner.png'))
                outer_path = os.path.join(outer_patches,
                                          (im_code + '_outer.png'))
            elif source is 'synth':
                # always use first synth for surrounds...
                inner_path = os.path.join(inner_synths,
                                          (im_code + '_inner_synth_1.png'))
                outer_path = os.path.join(outer_synths,
                                          (im_code + '_outer_synth_1.png'))

            inner_out_path = os.path.join(out_dir,
                                          (im_code + '_inner_' + source +
                                           '.png'))

            outer_out_path = os.path.join(out_dir,
                                          (im_code + '_outer_' + source +
                                           '.png'))

            do_patch(inner_path, inner_out_path, params['inner'])
            do_patch(outer_path, outer_out_path, params['outer'])

        # append details to master_dat:
        master_dat = master_dat.append(dat.iloc[idx, :], ignore_index=True)

    else:
        print('At least one patch failed to synthesise at size '
              + str(dat.ix[idx, 'new_size']))
        print(im_code)
        fail_count += 1

print('Number of failed patches: ' + str(fail_count))

# remove and rename columns in master dat to match other files:
master_dat.drop('size', axis=1, inplace=True)
master_dat.rename(columns={'new_size': 'size'}, inplace=True)

# save datafile:
master_dat.to_csv(os.path.join(out_dir, 'patch_info.csv'), index=False)

pu.files.session_info()
