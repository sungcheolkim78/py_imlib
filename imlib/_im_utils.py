#!/usr/local/bin/python3
"""
_im_utils.py - separate utility functions for tif images

date: 20160429
date: 20170810 - combine all util functions
date: 20180218 - add sift functions
date: 20180315 - tidy up and add Bobby's fitting algorithm
"""

import os
import tifffile
import skimage
import scipy, scipy.signal
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Sung-Cheol Kim'
__version__ = '1.2.2'


############################################ file processing
def readTif(filename, method='full'):
    """ old method to read tif using tifffile """
    print('... read file: %s' % filename)

    ffilename = filename.replace('BIOSUB', 'sBIOSUB')
    if method != 'full' and os.path.exists(ffilename):
        with tifffile.TiffFile(ffilename) as imfile:
            images = imfile.asarray()
            imgMean = skimage.img_as_float(images)
    elif os.path.exists(filename):
        with tifffile.TiffFile(filename) as imfile:
            images = imfile.asarray()
            images = skimage.img_as_float(images)
        if images.shape[0] > 2:
            imgMean = images.mean(axis=0)
        else:
            imgMean = images
        scipy.misc.imsave(ffilename, imgMean)
        print('... save mean image: %s' % ffilename)
    else:
        print('... no file')
        return 0

    if method == 'full':
        return images
    return imgMean


def _plotAxis(image, axis=0, background=0, backstd=0, window=None):
    '''
    plot line profile collectively from 2d image

    Inputs:
        image = 2D numpy array image
    Options:
        axis = x or y axis
        background = background value
        backstd = background standard deviation
        window =
    Return:
    '''

    if window is None:
        i0 = 0
        i1 = image.shape[axis]
    else:
        i0 = window[0]
        i1 = window[1]

    if axis in [0, 1]:
        if axis == 0:
            off_axis = 1
            plt.xlabel('x (pixels)')
            line = image[i0:i1, :].mean(axis=axis)
        else:
            off_axis = 0
            plt.ylabel('y (pixels)')
            line = image[:, i0:i1].mean(axis=axis)

    for i in range(i0, i1):
        if axis == 0:
            plt.plot(_smooth(image[i, :]), alpha=0.5)
        elif axis == 1:
            plt.plot(_smooth(image[:, i]), alpha=0.5)

    # pattern recognition 1
    localmaxs = scipy.signal.argrelmax(_smooth(line), order=10)
    for lm in localmaxs[0]:
        print("... local maximums: %i " % lm)

    # pattern recognition 2
    dline = _smooth(line[1:] - line[:-1])
    localmaxds = scipy.signal.argrelmax(np.abs(dline), order=15)
    for ldm in localmaxds[0]:
        print("... local maximum derivatives: %i " % ldm)

    # pattern recognition 3
    der_max = np.argmax(dline)
    der_min = np.argmin(dline)
    if (der_max+4 > i1):
        dI_max = line[der_max-3] - line[i1-1]
    elif (der_max-3 < 0):
        dI_max = line[0] - line[der_max+3]
    else:
        dI_max = line[der_max-3] - line[der_max+3]
    dI_min = line[der_min-3] - line[der_min+3]
    print("... maximum derivatives: %i dI %.2f " % (der_max, dI_max))
    print("... minimum derivatives: %i dI %.2f " % (der_min, dI_min))

    plt.plot(line, color='w')
    plt.plot(dline, color='gray')
    plt.xlim([0, image.shape[off_axis]])
    plt.vlines(localmaxs, 0.0, line[localmaxs], color='y', linestyles='dashed')
    plt.vlines(localmaxds, 0.0, line[localmaxds], color='b', linestyles='dashed')
    plt.hlines(background, 0, image.shape[off_axis], linestyles='dashed')
    plt.hlines(backstd+background, 0, image.shape[off_axis], linestyles='dashed')
    plt.show()


def showTwo(image1, image2):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121)
    im = plt.imshow(image1)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax1)

    ax2 = fig.add_subplot(122)
    divider2 = make_axes_locatable(ax2)
    im = plt.imshow(image2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax2)


############################################ image processing
def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


def blur_image(im, n, ny=None):
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im, g, mode='valid')
    return(improc)


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """ An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    using the numerical procedure presented in eq (11) A. Chambolle (2005).

    Input: noisy input image (grayscale), initial guess for U, weight of
    the TV-regularizing term, steplength, tolerance for stop criterion.

    Output: denoised and detextured image, texture residual. """

    m, n = im.shape # size  of noisy image

    # initialize
    U = U_init
    Px = im # x-component to the dual field
    Py = im # y-component of the dual field
    error = 1

    while( error > tolerance):
        Uold = U

        # gradient of primal variable
        GradUx = np.roll(U, -1, axis=1) - U # x-component of U's gradient
        GradUy = np.roll(U, -1, axis=0) - U # y-component of U's gradient

        # update the dual variable
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = np.maximum(1, np.sqrt((PxNew**2 + PyNew**2)))

        Px = PxNew/NormNew  # update of x-component (dual)
        Py = PyNew/NormNew  # update of y-component (dual)

        # update the primal variable
        RxPx = np.roll(Px, 1, axis=1) # right x-translation of x-component
        RyPy = np.roll(Py, 1, axis=0) # right y-translation of y-component

        DivP = (Px-RxPx)+(Py-RyPy) # divergence of the dual field
        U = im + tv_weight*DivP # update of the primal variable

        # update of error
        error = np.linalg.norm(U-Uold)/np.sqrt(n*m);

    return U, im-U # denoised image and texture residual


############################################ line profile and fitting
def _smooth(x, window_len=11, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError('Window is on of', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman')

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(np.int(window_len/2)-1):-np.int(window_len/2)]


def _find_channel(image, window=11, compareRatio=0.2, minimumIntensity=0.1):
    """
    find_channel

    Parameters:
        image - image source
        window - smoothing window
        compareRatio - criteria for intensity variation of two walls
        minimumIntensity - boundary delta value of intensity

    Return:
    """

    # normalize image value
    image = skimage.exposure.rescale_intensity(image, in_range='image', out_range=(0.0, 1.0))

    # find channel wall
    height, width = image.shape
    x_mean_line = image.mean(axis=0)

    # check wall is too close to the frame edge
    edge_pixel = 15
    x_mean_line = x_mean_line[edge_pixel:-edge_pixel]
    dx_mean_line = _smooth(x_mean_line[1:] - x_mean_line[:-1], window_len=window)
    dx_max = np.argmax(dx_mean_line)
    dx_min = np.argmin(dx_mean_line)
    dx = np.int((window-1)/2)

    # check wall and frame boundary
    print(image.shape)
    if (dx_max+dx >= width-2*edge_pixel):
        dI_max = x_mean_line[width-2*edge_pixel-1] - x_mean_line[dx_max-dx]
    elif (dx_max-dx < 0):
        dI_max = x_mean_line[dx_max+dx] - x_mean_line[0]
    else:
        dI_max = x_mean_line[dx_max+dx] - x_mean_line[dx_max-dx]

    if (dx_min+dx >= width-2*edge_pixel):
        dI_min = x_mean_line[dx_min-dx] - x_mean_line[width-2*edge_pixel-1]
    elif (dx_min-dx < 0):
        dI_min = x_mean_line[0] - x_mean_line[dx_min+dx]
    else:
        dI_min = x_mean_line[dx_min-dx] - x_mean_line[dx_min+dx]

    # compare threshold for wall
    if np.abs(dI_max - dI_min) < compareRatio:
        if dx_max < dx_min:
            x0, x1 = dx_max+edge_pixel, dx_min+edge_pixel
        else:
            x0, x1 = dx_min+edge_pixel, dx_max+edge_pixel
        width = x1 - x0
        print("... find wall x0, x1 and width: %i, %i, %i" % (x0, x1, width))
    else:
        print("... fail to find channel wall")
        print("... dx_max: %i dI_max: %.3f" % (dx_max, dI_max))
        print("... dx_min: %i dI_min: %.3f" % (dx_min, dI_min))
        return 0

    # find different channel domain
    y_mean_line = image[:, x0:x1].mean(axis=1)
    dy_mean_line = _smooth(y_mean_line[1:] - y_mean_line[:-1], window_len=window)
    dy_max = np.argmax(dy_mean_line)
    dy_min = np.argmin(dy_mean_line)
    dy = np.int((window-1)/2)
    if (dy_max+dy >= height) or (dy_max-dy < 0):
        dy = 0
    dI_max = y_mean_line[dy_max+dy] - y_mean_line[dy_max-dy]
    dy = np.int((window-1)/2)
    if (dy_min+dy >= height) or (dy_min-dy < 0):
        dy = 0
    dI_min = y_mean_line[dy_min-dy] - y_mean_line[dy_min+dy]

    if np.abs(dI_max) > minimumIntensity:
        if np.abs(dI_min) > minimumIntensity:
            if dy_max > dy_min:
                y0, y1 = dy_min, dy_max
            else:
                y0, y1 = dy_max, dy_min
            print("... three channel area: %i, %i" % (y0, y1))
            return (x0, x1, y0, y1)
        else:
            y0 = dy_max
            print("... two channel area: %i" % y0)
            print("... dy_min: %i, dI_min: %.3f" % (dy_min, dI_min))
            return (x0, x1, y0)
    elif np.abs(dI_min) > minimumIntensity:
        y0 = dy_min
        print("... two channel area: %i" % y0)
        print("... dy_max: %i, dI_max: %.3f" % (dy_max, dI_max))
        return (x0, x1, y0)
    else:
        print("... only channel")
        print("... dy_max: %i, dI_max: %.3f" % (dy_max, dI_max))
        print("... dy_min: %i, dI_min: %.3f" % (dy_min, dI_min))
        return (x0, x1)


def func(x, a, b, c):
    return 1.0/(c+np.abs((x-b)/a))


def gaussian(k, x):
    """ gaussian function
    k - coefficient array, x - values """
    return (k[0]/(np.sqrt(2*np.pi)*k[2])) * np.exp(-(x-k[1])**2 /(2*k[2]**2)) + k[3]


def line(k, x):
    """ line function """
    return k[0]*x + k[1]


def loss(k, x, y, f, nu):
    """ optimization function
    k - coefficients
    x, y - values
    f - function
    nu - normalization factor """
    res = y - f(k, x)
    return np.sum(np.log(1 + res**2/nu))


def robust_gaussian_fit(x, y, nu=1.0, initial=[1.0, 0.0, 1.0, 0.0], verb=False):
    """ robust fit using log loss function """
    return scipy.optimize.fmin(loss, initial, args=(x, y, gaussian, nu), disp=verb)


def robust_line_fit(x, y, nu=1.0, initial=[1.0, 0.0], verb=False):
    """ robust fit using log loss function """
    return scipy.optimize.fmin(loss, initial, args=(x, y, line, nu), disp=verb)


def find_after(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    (idx, ) = np.where(a - a0 > 0)
    return a[idx.min()]


def find_before(a, a0):
    (idx, ) = np.where(a - a0 < 0)
    return a[idx.max()]


def zprofile(imgMean, location):
    width = 2
    temp = imgMean[location-width:location+width, :]
    return np.sum(temp, axis=0)/(2.0*width+1.0)


def _showThresh(image, percentage=0.1, smoothing_pixel=30):
    Y = np.arange(image.shape[0])
    loc = np.arange(image.shape[0])
    for i in range(image.shape[0]):
        xline = image[i, :]
        s_xline = _smooth(xline, window_len=smoothing_pixel)
        threshold = s_xline.min() + percentage*(s_xline.max()-s_xline.min())
        loc[i] = np.abs(s_xline - threshold).argmin()
    plt.plot(loc, Y, color='white')
    # print(loc)
    plt.imshow(image)
    plt.show()


############################################ pattern matching
def process_image(img, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """ Process an image and save the results in a file. """

    try:
        import cv2
    except ImportError:
        print('... opencv is not installed')
        return

    #print(img.shape, img.min(), img.max())
    img = np.uint8(255.0*(img - img.min())/(img.max() - img.min()))

    # sift
    #sift = cv2.xfeatures2d.SIFT_create()
    #kp, desc = sift.detectAndCompute(img, None)

    # star + brief
    #star = cv2.xfeatures2d.StarDetector_create()
    #brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    #kp = star.detect(img, None)
    #kp, desc = brief.compute(img, kp)

    # ORB
    orb = cv2.ORB_create()
    kp, desc = orb.detectAndCompute(img, None)

    locs = []
    for k in kp:
        locs.append([k.pt[0], k.pt[1], k.size, k.angle])
    locs = np.array(locs)
    locs = locs.reshape((len(kp), 4))
    #print(locs.shape, desc.shape)
    write_features_to_file(resultname, locs, desc)


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """

    f = np.loadtxt(filename)
    return f[:, :4], f[:, 4:]


def write_features_to_file(filename, locs, desc):
    """ Save feature location and descriptor to file. """
    np.savetxt(filename, np.hstack((locs, desc)))


def plot_features(im, locs, circle=False):
    """ Show image with features. input: im (image as array),
    locs (row, col, scale, orientation of each feature). """

    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        plt.plot(x, y, 'b', linewidth=2)

    plt.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:, 0], locs[:, 1], 'ob')
    plt.axis('off')


def match(desc1, desc2):
    """ For each descriptor in the first image,
    select its match in the second image.
    input: desc1 (descriptors for the first image),
    desc2 (same for second image). """

    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = np.zeros((desc1_size[0],1), 'int')
    desc2t = desc2.T  # precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:], desc2t)   # vector of dot products
        dotprods = 0.9999*dotprods
        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))

        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio*np.arccos(dotprods)[indx[1]]:
            matchscores[i] = np.int(indx[0])

    return matchscores


def match_twosided(desc1, desc2):
    """ Two-sided symmetric version of match(). """

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12


def compute_harris_response(im, sigma=3):
    """ Compute the Harris corner detector response function
    for each pixel in a graylevel image. """

    from scipy.ndimage import filters
    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """ Retrun corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. """

    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T

    # ... and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0]-min_dist):(coords[i, 0]+min_dist),
                              (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """ Plots corners found in image. """

    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()


def get_descriptors(image, filtered_coords, wid=5):
    """ For each point return, pixel values around the point
    using a neighbourhood of width 2*wid+1. (Assume points are
    extracted with min_distance > wid). """

    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
                      coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)

    return desc


def match_h(desc1, desc2, threshold=0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using normalized cross-correlation. """

    n = len(desc1[0])

    # pair-wise distances
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = np.argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores


def match_twosided_h(desc1, desc2, threshold=0.5):
    """ Two-sided symmetric version of match(). """

    matches_12 = match_h(desc1, desc2, threshold)
    matches_21 = match_h(desc2, desc1, threshold)

    ndx_12 = np.where(matches_12 >= 0)[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12


def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.

    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1, im2 (images as arrays), locs1, locs2 (feature locations),
    matchscores (as output from 'match()'),
    show_below (if images should be shown below matches). """

    im3 = appendimages(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))

    plt.imshow(im3)

    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            plt.plot([locs1[i][1], locs2[m][1]+cols1], [locs1[i][0], locs2[m][0]], 'c')
    plt.axis('off')
    plt.show()


def ROF_values(f, x, y, clambda):
    """ Compute the ROF cost functional """
    a = np.linalg.norm((f-x).flatten())**2/2
    b = np.sum(np.sqrt(np.sum(y**2, axis=2)).flatten())
    return a + clambda*b

def prox_project(clambda, z):
    """ Projection to the clambda-ball """
    nrm = np.sqrt(z[:, :, 0]**2 + z[:, :, 1]**2)
    fact = np.minimum(clambda, nrm)
    fact = np.divide(fact, nrm, out=np.zeros_like(fact), where=nrm!=0)

    y = np.zeros(z.shape)
    y[:, :, 0] = np.multiply(z[:, :, 0], fact)
    y[:, :, 1] = np.multiply(z[:, :, 1], fact)
    return y

def projectedGD_ROf(image, clambda, iters=100):
    """ 2D Dual ROF solver using Projected Gradient Descent Method """
    start_time = timeit.default_timer()
    op = operators.make_finite_differences_operator(image.shape, 'fn', 1)
    y = op.val(image)
    x = image
    vallog = np.zeros(iters)
    alpha = 0.1

    for i in range(iters):
        y -= alpha * op.val(op.conj(y) - image)
        y = operators.prox_project(clambda, y)
        x = image - op.conj(y)
        vallog[i] = ROF_value(image, x, op.val(x), clambda)

    print("...Finished in %d iterations and %f sec" % (iters, timeit.default_timer() - start_time))
    return (x, vallog)

# vim:foldmethod=indent:foldlevel=0
