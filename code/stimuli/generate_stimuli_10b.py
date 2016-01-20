# coding: utf-8

"""Second step of stimulus generation for experiment 10. This script takes
the source patches and synths, windows them, resizes them,
if appropriate, before saving to final_ims.

This script assumes you've run `generate_stimuli_10a.py`, then

`p_s_generation_expt_10_....m`

To understand what this script is doing, see expt_10_stimulus_dev.ipynb.

Tom Wallis wrote it.

"""

##### Import functions that actually do the work ####

import os
import yaml
import pandas as pd
import helpers
import numpy as np
import psyutils as pu
from skimage import io, color, img_as_float, transform

"""Params for stimulus generation """

with open('generation_params_exp_10.yaml', 'r') as f:
    params = yaml.load(f)

# params is a dictionary containing the parameter values for stimulus gen.

# set up directories:
top_dir = helpers.project_directory()

source_dir = os.path.join(top_dir, 'stimuli', 'experiment-10',
                          'source_patches')
synth_dir = os.path.join(top_dir, 'stimuli', 'experiment-10',
                         'synths')
out_dir = os.path.join(top_dir, 'stimuli', 'experiment-10',
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
        im = transform.resize(im, (desired_size,
                                   desired_size))

    im = window_im(im)  # returns an rgba float.

    # save:
    io.imsave(out_name, im)


""" For each desired size, resize patch and synths if required.
Alpha blend with background ims, leaving a grey border..."""

dat = pd.read_csv(os.path.join(source_dir, 'patch_info.csv'))

# want to resize 100 of each image downwards
size_vec = np.repeat(params['patch_sizes'],
                     len(dat) / len(params['patch_sizes']))

fail_count = 0
# now we know the size and desired size, loop over ims,
# resizing if necessary, then window and save:
for idx, size in enumerate(size_vec):
    print('Scene ' + str(idx))
    im_code = dat.ix[idx, 'filename']

    # check that all files exist, or skip this image (in case patch can't be
    # synthed in some images):
    middle_synth_1 = os.path.join(synth_dir,
                                  (im_code + '_synth_1.png'))
    middle_synth_2 = os.path.join(synth_dir,
                                  (im_code + '_synth_2.png'))
    middle_synth_3 = os.path.join(synth_dir,
                                  (im_code + '_synth_3.png'))
    middle_synth_4 = os.path.join(synth_dir,
                                  (im_code + '_synth_4.png'))
    middle_synth_5 = os.path.join(synth_dir,
                                  (im_code + '_synth_5.png'))

    if os.path.exists(middle_synth_1) and \
            os.path.exists(middle_synth_2) and \
            os.path.exists(middle_synth_3) and \
            os.path.exists(middle_synth_4) and \
            os.path.exists(middle_synth_5):

        # natural target patch:
        in_path = os.path.join(source_dir, (im_code + '.png'))
        out_path = os.path.join(out_dir,
                                (im_code + '_natural.png'))
        dat.ix[idx, 'size'] = size

        do_patch(in_path, out_path, size)

        # synthesized patches:
        for synth in range(5):
            synth += 1
            in_path = os.path.join(synth_dir,
                                   (im_code + '_synth_' + str(synth) + '.png'))

            out_path = os.path.join(out_dir,
                                    (im_code + '_synth_' + str(synth) + '.png'))
            do_patch(in_path, out_path, size)

        # append details to master_dat:
        master_dat = master_dat.append(dat.iloc[idx, :], ignore_index=True)

    else:
        print('At least one patch failed to synthesise at size '
              + str(size))
        print(im_code)
        fail_count += 1

print('Number of failed patches: ' + str(fail_count))

# save datafile:
master_dat.to_csv(os.path.join(out_dir, 'patch_info.csv'), index=False)

pu.files.session_info()
