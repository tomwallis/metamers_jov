# coding: utf-8

import numpy as _np
import psyutils as _pu
from skimage import io as _io
from skimage import exposure as _exposure
from skimage import img_as_float as _img_as_float
from skimage import color as _color
import os as _os
import glob as _glob
from scipy.signal import convolve as _convolve


"""This file contains a number of common helper functions that will be called
by other stuff.

I suggest you import it into the shorthand namespace hlp::
    import helper_functions as hlp


Tom Wallis wrote it.

"""


def image_list(source_dir, image_format='.png', exclude_pilot=True):
    """ Return a list of filenames for the images in the source database.

    e.g. source_dir = '/Users/tomwallis/Dropbox/Image_Databases/kienzle'

    """

    # get a list of all the scenes in the directory:
    file_list = []
    # for file in glob.glob(os.path.join(source_dir, "*.png")):
    wildcard = '*' + image_format
    for file in _glob.glob(_os.path.join(source_dir, wildcard)):

        if exclude_pilot is True:
            if check_im_id(file) is False:
                file_list.append(file)
        else:
            file_list.append(file)

    return(file_list)


def list_mit_ims(source_path):
    """return two lists: first absolute paths to the MIT 1003 database
    images, second their corresponding fixation maps.

    :param source_path:
        The path to the MIT 1003 dataset root directory. Should have
        subdirectories of ALLFIXATIONMAPS and ALLSTIMULI.

    """

    img_dir = _os.path.join(source_path, 'ALLSTIMULI')
    fix_dir = _os.path.join(source_path, 'ALLFIXATIONMAPS')

    # list of all the scenes:
    img_list = []
    wildcard = '*.jpeg'
    for file in _glob.glob(_os.path.join(img_dir, wildcard)):
        img_list.append(file)

    fix_list = []
    wildcard = '*Map.jpg'
    for file in _glob.glob(_os.path.join(fix_dir, wildcard)):
        fix_list.append(file)

    return img_list, fix_list


def list_texture_ims(source_path):
    """return a list of absolute paths to the MIT texture database.

    :param source_path:
        The path to the MIT VisTex root directory.

    """

    img_dir = _os.path.join(source_path, 'FLAT', '512x512')

    # list of all the scenes:
    img_list = []
    wildcard = '*.ppm'
    for file in _glob.glob(_os.path.join(img_dir, wildcard)):
        img_list.append(file)

    return img_list


def load_im(fname):
    """ Load an image from a file, converting range 0--1 and a
    float.

    Returns:
        A numpy array containing the loaded image.
    """

    im = _io.imread(fname)
    im = _img_as_float(im)
    im = _exposure.rescale_intensity(im, out_range=(0, 1))
    return(im)


def load_fixation_image(fname, size):
    """Load a fixation map, crop it, return as numpy array with max
    1 and min 0.
    """
    im = _io.imread(fname)
    im = _img_as_float(im)
    im = _pu.image.cutout_patch(im, size)
    im = _exposure.rescale_intensity(im, out_range=(0, 1))
    return(im)


def crop_and_scale(im, size, rms):
    """ Crop an image to the appropriate size, scale in contrast and
    set mean. Currently just crops to centre of image.

    Returns:
        A numpy array containing the scaled image.

    """
    im = _pu.image.cutout_patch(im, size)
    im = _pu.image.contrast_image(im, sd=rms/2., returns="contrast")
    im += 0.5

    im[im < 0] = 0
    im[im > 1] = 1
    return(im)


def max_patch(im, filt_size):
    """Return the patch of maximum response (highest intensity)
    in the image. Might be used on an image containing fixation densities
    to determine where to crop the real image to have "maximum saliency".

    Args:
        im (2D matrix): the image to find the max response.

        filt_size (scalar): the filter size in pixels. Only set up
            for square filter regions, currently.

    Returns:
        max_im: a 2D array of size (filt_size, filt_size) containing
            the maximum response.
        max_x: a scalar containing the left-hand x coord.
        max_y: the top y coord of the max_im box.
    """

    filter_kernel = _np.ones((filt_size, filt_size))
    filter_responses = _convolve(im, filter_kernel, mode='valid')
    position_of_maxium = _np.unravel_index(filter_responses.argmax(),
                                           filter_responses.shape)
    max_y, max_x = position_of_maxium
    max_im = im[max_y:max_y+filt_size, max_x:max_x+filt_size]
    return(max_im, max_x, max_y)


def check_im_id(fname):
    """Check whether the image is one of the pilot images. Returns
    True if a pilot image, False otherwise.

    """

    im_code = _os.path.split(fname)[1][:5]
    pilot_im_id = ['00075', '00084', '00273', '00492', '00523']

    pilot_ims = False

    for i in pilot_im_id:
        if i == im_code:
            pilot_ims = True

    return(pilot_ims)


def project_directory():
    """Returns the full path of the top level directory for this project.

    Assuming that this function is being called from somewhere in the project.

    """

    orig_dir = _os.getcwd()
    parent = _os.path.split(orig_dir)[0]
    target = "metamers-natural-scenes"

    try:
        # go up directories, searching for target:
        while parent[-len(target):] != "metamers-natural-scenes":
            _os.chdir('..')
            parent = _os.path.split(_os.getcwd())[0]

        top_path = parent
        # set working directory back to original:
        _os.chdir(orig_dir)
    except:
        _os.chdir(orig_dir)
    return top_path


def return_patch_positions(eccent, size, pix_per_deg):
    """ return the (x, y) locations of the patch centres.

    Args:
        eccent: the eccentricity in degrees.
        size: the size of the patch in pixels.
        pix_per_deg: the number of pixels per degree

    Returns:
        a dictionary containing keys for locations 't', 'b',
        'r' and 'l'. The values are lists of two elements corresponding
        to the x (horizontal) and y (vertical) coordinates relative
        to a space starting at (0, 0) in the top left.
    """

    dim = (size, size)
    eccent_pix = eccent * pix_per_deg

    patch_positions = {'t': [dim[1]/2., dim[0]/2. - eccent_pix],
                       'b': [dim[1]/2., dim[0]/2. + eccent_pix],
                       'l': [dim[1]/2. - eccent_pix, dim[0]/2.],
                       'r': [dim[1]/2. + eccent_pix, dim[0]/2.]}
    return(patch_positions)


def cut_out_patch(im, position, patch_size,
                  eccent, im_size, pix_per_deg):
    """Return the patch corresponding to position in the image.

    """
    im = im.copy()
    patch_positions = return_patch_positions(eccent, im_size, pix_per_deg)
    centre_x, centre_y = patch_positions[position]

    im = im[centre_y - (.5*patch_size): centre_y + (.5*patch_size),
            centre_x - (.5*patch_size): centre_x + (.5*patch_size)]

    return(im, centre_x, centre_y)


def surround_window(im, win, position, size,
                    eccent, im_size, pix_per_deg):
    """Function to cut out a patch of an image and include a grey region
    running to zero. Also looks after border effects (when desired surround
    size exceeds image boundaries).

    Called by generate_background_ims
    """

    im = im.copy()
    patch_positions = return_patch_positions(eccent, im_size, pix_per_deg)
    centre_x, centre_y = patch_positions[position]

    # define patch coordinates:
    start_y = centre_y - (.5*size)
    end_y = centre_y + (.5*size)
    start_x = centre_x - (.5*size)
    end_x = centre_x + (.5*size)

    if start_y < 0:
        win = win[-start_y:, :]
        start_y = 0
    if end_y > im.shape[0]:
        eps = end_y - im.shape[0]
        win = win[0:-eps, :]
        end_y = im.shape[0]
    if start_x < 0:
        win = win[:, -start_x:]
        start_x = 0
    if end_x > im.shape[1]:
        eps = end_x - im.shape[1]
        win = win[:, 0:-eps]
        end_x = im.shape[1]

    patch = im[start_y: end_y,
               start_x: end_x]

    patch *= (1 - win)
    patch += win * 0.5

    return(im)


def make_teardrop(img_size, r0, a0, radial_size, aspect_ratio):
    """ Heiko's function to return a boolean mask
    containing a teardrop-shaped zone. The aspect ratio is measured by
    dividing the radial line and the largest "Kreisbogen" within the shape.

    Args:
        img_size:
            the size of the larger image containing the teardrop in pixels.
            Passed as a numpy array with two elements.
        r0:
            the radial centre of the zone (from the middle of the image).
        a0:
            angular centre of the zone (0 is right, np.pi/2 is top).
        radial_size:
            radial length of the patch in pixels.
        aspect_ratio:
            ratio of radial to angular extent. 1 = basically circular,
            2 is a teardrop.

    Example::
        m1 = make_teardrop(np.array([768, 768]), 256, np.pi/2, 256, 2)
        show_im(m1)

    """

    radial_size = radial_size/2  # from diameter to radius
    mask = _np.ones(img_size)
    center = _np.array(img_size-1, dtype='float_')/2
    x = range(img_size[0])-center[0]
    y = _np.transpose((range(img_size[1])-center[1])[_np.newaxis])
    radius = _np.sqrt(x*x+y*y)
    radius = radius-r0
    angle = _np.arctan2(x, y)
    angle = angle-a0
    angle[angle < -_np.pi] = 2 * _np.pi + angle[angle < -_np.pi]
        # to avoid problems at the -pi/pi border
    angular_size = radial_size/(aspect_ratio*r0)
    ellipse = _np.sqrt(radius**2 / (radial_size**2)
                       + angle**2 / (angular_size**2))
    mask = ellipse < 1
    return(mask)


def window_patch(im, inverted=False):
    """Return an image patch after windowing. Expects a 2D array (greyscale)
    containing unsigned floats (range 0--1) as input, and returns a 3D
    array with four planes (i.e. RGBA). The alpha
    channel contains a circular cosine ramp.

    Called by make_compound_images

    """

    win = _pu.image.cos_win_2d(size=im.shape[0], ramp=5, ramp_type='pixels')
    res = _np.ndarray((im.shape[0], im.shape[1], 4), dtype=_np.float)
    res[..., :3] = _color.gray2rgb(im)
    res[..., 3] = _np.ones_like(im)

    if inverted is True:
        res[..., 3] = res[..., 3] * (1 - win)
    else:
        res[..., 3] = res[..., 3] * win
    return(res)
