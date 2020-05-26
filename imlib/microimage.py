#!/usr/bin/env python
"""
microimage.py - microscopy images class

date: 20180317 - version 1.0.0 - first version, derived from tiffile class
date: 20180327 - version 1.1.0 - optimization and fast read
date: 20180405 - version 1.1.1 - add crop and save
date: 20180704 - version 1.2.0 - tune blob detection
date: 20180706 - version 1.2.1 - add denoise module
"""

import os
import sys
import glob
import pims
import pandas
import tqdm
import skimage
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import trackpy as tp

from sklearn.cluster import MiniBatchKMeans
from lmfit.models import LinearModel
from lmfit.models import GaussianModel
import skimage.restoration
import cv2

import imlib._im_utils as utils
from .rfit import robust_line_fit
from .rfit import robust_gaussian_fit
from .rfit import robust_gaussian2_fit
from .rfit import robust_gaussian3_fit
from .rfit import gaussian, gaussian2, gaussian3
from .noisegater2d import NoiseGater

__author__ = 'Sungcheol Kim <kimsung@us.ibm.com>'
__version__ = '1.2.1'


class mImage(object):

    def __init__(self, filename, update=False, method='pims', debug=False, nd2_m=0, nd2_c=1):
        """ TifFile class initialization """

        # set debug flag
        self._debug = debug

        # image information
        self._images = []
        self._width, self._height = [0, 0]
        self._frameN = 0
        self._frames = []
        self._exposuretime = 0.037
        self._mpp = 1.0     # meter per pixel
        self._magnification = 1.0
        self._nd2 = []
        self._nd2_m = nd2_m
        self._nd2_c = nd2_c
        self._newimages = []
        self._thimages = []

        # file information
        self._fname = self.open(filename, method=method)
        self._cwd = os.getcwd()
        self._curframe = 0

        # preprocessing information
        self._cmapname = 'viridis'
        self._blur = 0
        self._filter = 'none'
        self._updown = False
        self._im = None
        self._cnts = []
        self.box_arr_ = []

        # line fit information
        self._baseline = 0.0
        self._beamdirection = 'right'
        self._beaminfo = []
        self._peakdata = []
        self._beamvel = 50.0
        self._kfit = []
        self._wallinfo = []
        self._frameangle = 0.0

        self._duration = self._exposuretime * self._frameN

        # frame information
        self._imgMean = []
        self._imgMax = []
        self._imgMin = []
        self._range = [0, 0]

        if self._debug: print('... preload mean, max, min image - maybe slow')
        self.preload(update=update)

    def open(self, filename, method='pims'):
        """ read file """
        # set filename or find filename
        if not os.path.isfile(filename):
            flist = glob.glob('*%s*' % filename)
            if len(flist) == 0:
                print('... no file: %s' % filename)
                sys.exit(1)
            else:
                print('... find %i files: %s' % (len(flist), flist[0]))
                filename = flist[0]

        # read file depending on extension
        from slicerator import pipeline

        @pipeline
        def as_grey(frame):
            red = frame[:, :, 0]
            green = frame[:, :, 1]
            blue = frame[:, :, 2]
            return 0.2125 * red + 0.7154 * green + 0.0721 * blue

        ext = filename.split('.')[-1]

        # avi format
        if ext == 'avi':
            # with pims.PyAVReaderIndexed(filename) as tif:
            if self._debug: print('... read %s with moviepy library' % filename)
            with pims.MoviePyReader(filename) as tif:
                self._images = as_grey(tif)
                self._width, self._height = self._images.frame_shape[:-1]
                self._frameN = len(self._images)
        # using bioformat library
        elif (method == 'bioformat'):
            if self._debug: print('... read %s with bioformat library' % filename)
            with pims.Bioformats(filename) as tif:
                self._images = tif
                meta = tif.metadata
                self._frameN = meta.ImageCount()
                self._width = meta.PixelsSizeX(0)
                self._height = meta.PixelsSizeY(0)
                self._exposuretime = meta.PlaneExposureTime(0, 0)
                self._mpp = meta.PixelsPhysicalSizeX(0)
        # nd2 format
        elif ext == 'nd2':
            if self._debug: print('... read %s with nd2_reader library' % filename)
            with pims.ND2_Reader(filename) as tif:
                #print(tif.sizes)
                if 't' in tif.sizes:
                    tif.bundle_axes = 'txy'
                if 'm' in tif.sizes:
                    if tif.sizes['m'] > 1:
                        tif.iter_axes = 'm'
                        print('... multistack image %i of %i selected' % (self._nd2_m + 1, tif.sizes['m']))
                if 'c' in tif.sizes:
                    tif.default_coords['c'] = self._nd2_c
                    print('... multichannel image %i of %i selected' % (self._nd2_c + 1, tif.sizes['c']))

                self._nd2 = tif
                self._images = tif[self._nd2_m]
                self._width, self._height = tif.sizes['x'], tif.sizes['y']
                self._frameN = tif.sizes['t']
                self._planeN = tif.sizes['m']
                self._mpp = tif.metadata['calibration_um']
                self._exposuretime = tif[0].metadata['t_ms']

        # tif format with tifffile library
        elif (ext == 'tif') and (method == 'tifffile'):
            if self._debug: print('... read %s with tifffile library' % filename)
            with tifffile.TiffFile(filename) as tif:
                imgs = tif.asarray()
                # self._images = np.swapaxes(imgs, 0, 2)
                self._images = pims.Frame(imgs)
                self._width, self._height = self._images.shape[1:]
                self._frameN = self._images.shape[0]
        # other format
        else:
            with pims.open(filename) as tif:
                if len(tif.frame_shape) == 2:    # grey image
                    if self._debug: print('... read %s with pims library - black/white' % filename)
                    self._images = tif
                    self._width, self._height = self._images.frame_shape
                else:   # color image
                    if self._debug: print('... read %s with pims library - converted to b/w' % filename)
                    self._images = as_grey(tif)
                    self._width, self._height = self._images.frame_shape[:-1]
                self._frameN = len(self._images)

        if self._frameN == 1:
            self._frames = [0]
        else:
            self._frames = range(self._frameN)

        return filename

    def preload(self, update=False):
        """ load mean, max, min, mp4 files """

        # check previous analysis
        self._imgMean = self.mean(update=update)
        self._imgMax = self.max(update=update)
        self._imgMin = self.min(update=update)
        if update:
            self._range = (np.min(self._imgMin), np.max(self._imgMax))

    def __repr__(self):
        """ representation of TifFile """
        msg = ''
        msg += '... File name: %s\n' % self._fname
        msg += '... frames: %i\n' % self._frameN
        msg += '... frame size: (%i, %i)\n' % (self._width, self._height)
        msg += '... range: (%f, %f)\n' % (self._range[0], self._range[1])
        msg += '... colormap: %s\n' % self._cmapname
        msg += '... blur: %i [pixels]\n' % self._blur
        msg += '... equlaize filter: %s\n' % self._filter
        if self._exposuretime > 0:
            msg += '... exposure time: %f\n' % self._exposuretime
        if self._mpp != 1.0:
            msg += '... meter per pixel: %f\n' % self._mpp
        return msg

    # data conversion
    def asframe(self):
        """ return pims Frame object """

        # _imgages vs _newimages - raw data and processed data
        if len(self._newimages) > 0:
            return pims.Frame(self._newimages)
        else:
            return pims.Frame(self._images)

    def asarray(self):
        """ return numpy array for frames """
        if len(self._newimages) > 0:
            return np.array(self._newimages[:])
        else:
            return np.array(self._images[:])

    def getframe(self, frame=-1, types='pims', orig=False):
        """ get frame """
        if frame == -1:
            frame = self._curframe
        if not (frame in self._frames):
            print('... no frame %i - max frame: %i' % (frame, self._frameN))
            return False

        if orig:
            img = self._images[frame]
        elif len(self._newimages) > 0:
            img = self._newimages[frame]
        else:
            img = self._images[frame]

        if types == 'uint8':
            img = img.astype(np.float32)
            img = (img - img.min()) / img.ptp()
            return np.array(img * 255., dtype=np.uint8)
        elif types == 'uint16':
            img = skimage.exposure.rescale_intensity(img, in_range='image', out_range='uint16')
            return np.uint16(img)
        elif types == 'float':
            img = skimage.img_as_float(img)
            return skimage.exposure.rescale_intensity(img, in_range='image', out_range=(0.0, 1.0))
        else:
            return pims.Frame(img)

    def mean(self, scale=-1, update=False):
        """ calculate frame mean image """

        filename = self._fname[:-4] + '_mean.png'
        if (not update):
            if len(self._imgMean) > 0:
                img = self._imgMean
            elif os.path.isfile(filename):
                img = pims.open(filename, as_gray=True)[0]
                if self._debug: print('... read from %s' % filename)
            else:
                update = True
        if update:
            if self._debug: print('... calculate mean')
            if self._frameN == 1:
                img = self._images[0]
            else:
                img = np.mean(self._images, axis=0)
                # normalize
                img = (img - img.min()) / (img.max() - img.min())
                plt.imsave(fname=filename, arr=img)

        self._imgMean = img
        self._range = (np.min(img), np.max(img))

        if scale > 0.0: img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        if self._filter is not 'none':
            return self._preprocess(img)
        else:
            return pims.Frame(img)

    def max(self, update=False):
        """ calculate frame maximum image """

        filename = self._fname[:-4] + '_max.png'
        if (not update):
            if len(self._imgMax) > 0:
                img = self._imgMax
            elif os.path.isfile(filename):
                img = pims.open(filename, as_gray=True)[0]
                if self._debug: print('... read from %s' % filename)
            else:
                update = True
        if update:
            if self._debug: print('... calculate mean')
            if self._frameN == 1:
                img = self._images[0]
            else:
                img = np.max(self._images, axis=0)
                plt.imsave(fname=filename, arr=img)

        self._imgMax = img
        self._range = (np.min(img), np.max(img))

        if self._filter is not 'none':
            return self._preprocess(img)
        else:
            return pims.Frame(img)

    def min(self, update=False):
        """ calculate frame minimum image """

        filename = self._fname[:-4] + '_min.png'
        if (not update) and os.path.isfile(filename):
            return pims.open(filename, as_gray=True)[0]

        if update or (len(self._imgMin) <= 1):
            if self._frameN == 1:
                img = self.getframe()
            else:
                img = np.min(self._images, axis=0)
                plt.imsave(fname=filename, arr=img)
        else:
            img = self._imgMin

        if self._filter is not 'none':
            return self._preprocess(img)
        else:
            return pims.Frame(img)

    def crop(self, margin=30, min_size=30, erode_iter=2, verb=False, box_arr=[]):
        """ crop image for all frames """

        if verb:
            print('... width, height: %i, %i' % (self._width, self._height))

        # check given box_arr
        if len(box_arr) == 0:
            th = self.threshold(0, method='otsu', erode_iter=erode_iter, show=False)
            cnt_data = self.find_contour(0, threshold=th, show=verb, min_size=min_size)
            d = np.asarray(cnt_data)
            box_arr = d[:, 2:6]
        else:
            box_arr = [box_arr]

        # show image and box area
        ax = self._showimage(self.mean())
        for i, b in enumerate(box_arr):
            [x0, y0, x1, y1] = np.array(b) + np.array([-margin, -margin, margin, margin])

            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(self._height-1, x1), min(self._width-1, y1)

            x = [x0, x1, x1, x0, x0]
            y = [y0, y0, y1, y1, y0]
            ax.plot(x, y, '--', color='gray', alpha=0.8)
            ax.annotate(str(i), xy=(x0, y0), color='white')
            if verb:
                print("... [{}] ({}, {}, {}, {})".format(i, x0, y0, x1, y1))
        plt.show()

        # save cropped area
        a = input("Confirm? [Yes|No]")
        if a.upper() in ["NO", "N"]:
            return False
        elif a.upper() in ["YES", "Y"]:
            self.save_tif(box_arr=box_arr, margin=margin)
            return True
        else:
            return False

    # set plot parameters
    def setFilter(self, filtername):
        """ set normalization filter
        Args:
            filtername - 'contrast', 'equalization', 'adapt'
        """
        flist = ['none', 'contrast', 'equalization', 'adapt']
        if filtername in flist:
            self._filter = filtername
        else:
            print('... filter list: ', flist)

    # image processing
    def _preprocess(self, obj=-1):
        """ preprocess image mostly by renormalization """
        if isinstance(obj, int):
            if obj == -1:
                obj = self._curframe
            elif obj in self._frames:
                self._curframe = obj
            else:
                return False
            img = skimage.img_as_float(self._images[obj])
        else:
            img = obj

        # histogram filter
        if self._filter == 'contrast':
            p2, p98 = np.percentile(img, (2, 98))
            img = skimage.exposure.rescale_intensity(img, in_range=(p2, p98))
        elif self._filter == 'equalization':
            img = skimage.exposure.equalize_hist(img)
        elif self._filter == 'adapt':
            img = np.uint16(skimage.exposure.rescale_intensity(img, in_range='image', out_range=(0, 255)))
            img = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)
        elif self._filter == 'gamma':
            img = skimage.exposure.adjust_gamma(img, gamma=0.25)
        elif self._filter == 'sigmoid':
            img = skimage.exposure.adjust_sigmoid(img)

        if skimage.exposure.is_low_contrast(img):
            print('... low contrast image')

        # blur filter
        if self._blur > 0:
            img = cv2.GaussianBlur(img, (self._blur, self._blur), 0)

        # image manipulation
        if self._updown:
            img = np.flipud(img)

        self._range = (img.min(), img.max())

        return pims.Frame(img)

    def denoise(self, frame=-1, method='fastNl', intensity=-1, show=False, save=False, **keywords):
        """ denoise frame """

        # check frame number
        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
        else:
            print('... {} is not in range.'.format(frame))
            return False

        # read image
        if method in ['rof', 'tvl', 'wavelet', 'deforest']:
            img = self.getframe(frame, types='float')
        else:
            img = self.getframe(frame, types='uint8')
        #    from bilevel_imaging_toolbox import solvers

        if method == 'rof':
            if intensity == -1: intensity = 70
            elif intensity == 'full': intensity = 10
            #U, v = solvers.chambolle_pock_ROF(img, 8, 0.25, 0.5, intensity*10)
            U = skimage.restoration.denoise_tv_bregman(img, intensity)
            U = (U * 255.0).astype(np.uint8)
        elif method == 'tvl':
            if intensity == -1: intensity = 0.01
            elif intensity == 'full': intensity = 0.05
            #U, v = solvers.chambolle_pock_TVl1(img, 8, 0.25, 0.5, intensity*10)
            U = skimage.restoration.denoise_tv_chambolle(img, intensity)
            U = (U * 255.0).astype(np.uint8)
        elif method == 'wavelet':
            if intensity == -1: intensity = 1
            elif intensity == 'full': intensity = 3
            U = skimage.restoration.denoise_wavelet(img, wavelet_levels=intensity, **keywords)
            U = (U * 255.0).astype(np.uint8)
        elif method == 'deforest':
            if intensity == -1: intensity = 3.0
            elif intensity == 'full': intensity = 3.0
            ng0 = NoiseGater(img, gamma=intensity, **keywords)
            U = ng0.clean()
            U = (U * 255.0).astype(np.uint8)
        elif method == 'bilateral':
            if intensity == -1: intensity = 10
            elif intensity == 'full': intensity = 75
            U = cv2.bilateralFilter(img, 5, intensity, intensity)
        elif method == 'gaussian':
            if intensity == -1: intensity = 3
            elif intensity == 'full': intensity = 7
            U = cv2.GaussianBlur(img, (intensity, intensity), 0)
        elif method == 'median':
            if intensity == -1: intensity = 3
            elif intensity == 'full': intensity = 7
            U = cv2.medianBlur(img, intensity)
        else:
            if intensity == -1: intensity = 5
            elif intensity == 'full': intensity = 9
            U = cv2.fastNlMeansDenoising(img, h=intensity, **keywords)

        self._range = (U.min(), U.max())

        if show:
            self._showimage(U)
            plt.show()

        if save:
            if len(self._newimages) == 0:
                self._newimages = np.array(self._images[:])
            else:
                self._newimages[frame] = U

        return pims.Frame(U)

    def denoise_all(self, method='fastNl', intensity=-1, refresh=False, **keywords):
        """ denoise all frames """

        if refresh:
            self._newimages = np.array(self._images[:])

        self.denoise(frame=0, method=method, show=False, save=True, intensity=intensity, **keywords)
        for f in tqdm.tqdm(self._frames):
            self.denoise(frame=f, method=method, show=False, save=True, intensity=intensity, **keywords)

    def sharpen(self, frame=-1, amount=0.3, show=False, save=False):
        """ sharpen frame """
        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
            img = self.getframe(frame, types='uint8')
            if self._blur == 0: self._blur = 3
            img_g = cv2.GaussianBlur(img, (self._blur, self._blur), 0)
            U = img + (img - img_g) * amount
            Umin = U.min()
            if Umin < 0:
                U -= Umin
            self._range = (U.min(), U.max())

            if show:
                self._showimage(U)
                plt.show()

            if save:
                if len(self._newimages) == 0:
                    self._newimages = np.array(self._images[:])
                else:
                    self._newimages[frame] = U

            return pims.Frame(U)

    def sift(self, frame=-1, show=True):
        """ calculate features using sift in cv2 """
        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
            img = self.getframe(frame)
            sift_fname = self._fname[:-4] + str(frame) + '.sift'
            utils.process_image(img, sift_fname)
            l1, d1 = utils.read_features_from_file(sift_fname)
            if show:
                utils.plot_features(img, l1, circle=True)

            return l1, d1

    def matches(self, frame1=-1, frame2=-1):
        """ find matched features """
        if frame1 == -1:
            frame1 = self._curframe
            if frame2 == -1:
                frame2 = frame1 + 1
            else:
                if frame2 in self._frames:
                    print('... matches frame %i and %i' % (frame1, frame2))
                else:
                    frame2 = frame1 + 1
        if frame1 in self._frames:
            self._curframe = frame1
            if frame2 == -1:
                frame2 = frame1 + 1
            else:
                if frame2 in self._frames:
                    print('... matches frame %i and %i' % (frame1, frame2))
                else:
                    frame2 = frame1 + 1

        img1 = self.getframe(frame=frame1)
        img2 = self.getframe(frame=frame2)
        img1 = np.uint8(255.0 * (img1 - img1.min()) / (img1.max() - img1.min()))
        img2 = np.uint8(255.0 * (img2 - img2.min()) / (img2.max() - img2.min()))
        # orb = cv2.ORB_create()
        # kp1, des1 = orb.detectAndCompute(img1, None)
        # kp2, des2 = orb.detectAndCompute(img2, None)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # l1, d1 = self.sift(frame=frame1, show=False)
        # l2, d2 = self.sift(frame=frame2, show=False)

        print('... starting matching')
        # match_res = utils.match_twosided(des1, des2)
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher()
        # match_res = bf.match(des1, des2)
        match_res = bf.knnMatch(des1, des2, k=2)
        # match_res = sorted(match_res, key = lambda x:x.distance)
        good = []
        for m, n in match_res:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        print('... %i match points' % len(match_res))
        img3 = np.zeros_like(np.hstack((img1, img2)))
        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, match_res[:20], img3, flags=2)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3, flags=2)
        plt.imshow(img3)
        # utils.plot_matches(self.getframe(frame1), self.getframe(frame2), l1, l2, match_res)
        plt.axis('off')
        plt.show()

    def threshold(self, frame=-1, method='otsu', erode_iter=0, show=True, save=False, img=None):
        """ automatic threshold finder """

        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
        else:
            print('... {} is not in range'.format(frame))
            return False

        if img is None:
            U = self.getframe(frame, types='uint8')
        else:
            U = img

        # find threshold using otsu
        if method == 'otsu':
            ret, th = cv2.threshold(U, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if method == 'adaptive':   # not working well
            th = cv2.adaptiveThreshold(U, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            ret = 0
        else:
            kth, centers = self.kmean(frame, n_cluster=2)
            ret = np.mean(centers)
            th = np.zeros_like(kth, dtype='uint8')
            th[kth > ret] = 255

        # morphological transformation
        if erode_iter > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            th = cv2.erode(th, kernel, iterations=erode_iter)
            th = cv2.dilate(th, kernel, iterations=erode_iter)

        if show:
            M = cv2.moments(th)
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            area = M['m00']

            fig = plt.figure(figsize=(10,5))
            plt.imshow(np.hstack((U, th)))
            plt.axis('off')
            plt.show()
            print('...thresh: %i, Area: %i' % (ret, area))
            print('...center: (%i, %i)' % (cx, cy))

        if save:
            self._tmp = th
            return None
        else:
            return th

    def detect_drift(self, frames=-1, min_size=200, show=True, save=False, erode_iter=1, full=True):
        """ detect drift by c.o.m """
        if frames == -1:
            frames = range(self._frameN)
        if save and (len(self._newimages) == 0):
            self._newimages = np.array(self._images[:])

        # iterate through all frames
        xy = np.zeros((len(frames), 2))
        for i in tqdm.tqdm(range(len(frames))):

            # find contour of cells
            th = self.threshold(frame=frames[i], method='otsu', erode_iter=erode_iter, show=False)
            if full:
                M = cv2.moments(th)
                c = [M['m10']/M['m00'], M['m01']/M['m00']]
            else:
                cnt_data = self.find_contour(frame=frames[i], threshold=th, show=False, min_size=min_size)
                c = cnt_data['cx'].iloc[0], cnt_data['cy'].iloc[0]
            xy[i, 0] = c[0]
            xy[i, 1] = c[1]

            # calculate shift and modify images
            if save:
                shift_x = c[0] - xy[0,0]
                shift_y = c[1] - xy[0,1]
                M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
                img = self.getframe(i, types='uint8')
                shifted_img = cv2.warpAffine(img, M, (self._height, self._width), None, cv2.INTER_CUBIC, cv2.BORDER_WRAP)
                self._newimages[i, :, :] = shifted_img

        # save shift coordinate in panda dataframe
        result = pd.DataFrame(xy, index=frames, columns=['x', 'y'])

        if show:
            plt.imshow(self.getframe(frames[0]))
            plt.plot(xy[:, 0], xy[:, 1], 'white')
            plt.show()

        return result

    def remove_cell(self, frame=-1, percentile=0.1, show=True):
        """ find condensate using double threshold """

        # check frame number
        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
        else:
            print('... {} is not frame range'.format(frame))
            return False

        # use kmean to select foreground and background
        U = self.getframe(frame, types='uint8')
        #U = self.denoise(frame, intensity=15)
        th, center = self.kmean(frame, n_cluster=2, show=self._debug)

        # prepare divider value between foreground and max intensity
        background, foreground = np.min(center), np.max(center)
        divider = foreground + (U.max() - foreground) * percentile

        U2 = U.copy()
        U2[U2 < divider] = divider

        # histogram equalize for noise background
        U2 = cv2.equalizeHist(U2)

        if show:
            plt.imshow(np.hstack((U, U2)))
            plt.axis('off')
            plt.show()
            print('thresh1: %i, Area sum: %i' % (background, th.sum() / 255.0))
        #    print('thresh2: %i, Area sum: %i' % (divider, th2.sum() / 255.0))

        return U2

    def remove_cells(self, frames=-1, percentile=0.1, save=True):
        if frames == -1:
            frames = range(self._frameN)

        n_frameN = len(frames)
        imgs = np.zeros((n_frameN, self._width, self._height))
        for i in tqdm.tqdm(frames):
            U = self.remove_cell(frame=i, percentile=percentile, show=False)
            imgs[i, :, :] = U

        if save:
            self._thimages = imgs

        return pims.Frame(imgs)

    def detect_blob(self, frame=-1, method='trackpy', res=5, show=True, minmass=0):
        """ detect blob using opencv """

        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
        else:
            print('... {} is not in range'.format(frame))
            return False

        if len(self._thimages) > 0:
            U2 = self._thimages[frame, :, :]
        else:
            U2 = self.remove_cell(frame, show=show)

        if method == 'cv2':

            detector = cv2.SimpleBlobDetector_create()
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 25
            params.filterByCircularity = True
            params.minCircularity = 0.3
            params.filterByConvexity = True
            params.minConvexity = 0.87
            params.filterByInertia = True
            params.minInertiaRatio = 0.01
            detector = cv2.SimpleBlobDetector_create(params)

            kp = detector.detect(U2)

            if show:
                print('... %i points are found!' % len(kp))
                im_kp = cv2.drawKeypoints(th, kp, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # print(U.shape, im_kp.shape)
                plt.imshow(np.hstack((self.getframe(frame), im_kp[:, :, 0])))

                i = 0
                for k in kp:
                    plt.annotate(str(i), xy=(k.pt[0], k.pt[1]), xytext=(k.pt[0] + 21, k.pt[1] - 21), arrowprops=dict(arrowstyle="-|>", facecolor='white', edgecolor='white'), color='white')
                    i += 1
                    # plt.plot(k.pt[0], k.pt[1], 'w.')
                    print('(%f %f)' % (k.pt[0], k.pt[1]))
                plt.show()

            return kp
        else:
            f = tp.locate(U2, res, minmass=minmass)
            tp.annotate(f, self.getframe(frame))
            return f

    def detect_trace(self, res=5, minmass=100, length=5, psize=0.0, ecc=1.0, show=True, drift=[], mag=3.0, loc=1):
        """ detect particle tracks using trackpy and save as pt file """

        if len(self._thimages) == 0:
            self.remove_cells(save=True)

        # using trackpy to select particles
        f = tp.batch(self._thimages, res, minmass=minmass)
        t = tp.link_df(f, 5, memory=3)
        t1 = tp.filter_stubs(t, length)
        # select by size and eccentricity
        t2 = t1[((t1['size'] > psize) & (t1['ecc'] < ecc))]

        # adjust drift
        if len(drift) == 0:
            dx = np.zeros(len(self._frames))
            dy = np.zeros(len(self._frames))
        else:
            dx = drift['x'].values - drift['x'].values[0]
            dy = drift['y'].values - drift['y'].values[0]

        if show:
            plt.clf()
            pid = np.unique(t2['particle'])
            print('... %i trajectories' % len(pid))

            if loc == 1:
                x0 = self._height * 0.25
                y0 = self._width * 0.75
            elif loc == 2:
                x0 = self._height * 0.75
                y0 = self._width * 0.75
            elif loc == 3:
                x0 = self._height * 0.75
                y0 = self._width * 0.25
            elif loc == 4:
                x0 = self._height * 0.25
                y0 = self._width * 0.25
            br = 9.3 * mag * 1.5

            for i, p in enumerate(pid):
                # plot individual traces
                if i > 10:
                    continue
                idx = np.where(t2['particle'] == p)[0]
                fr = t2['frame'].iloc[idx]
                x = t2['x'].iloc[idx] - dx[fr]
                y = t2['y'].iloc[idx] - dy[fr]
                nx = mag * (x.values - x.iloc[0]) + x0
                ny = mag * (y.values - y.iloc[0]) + y0
                plt.plot(nx, ny, label='condensate %i (%i)' % (p, len(x)))
                plt.plot([x0 - br, x0 + br, x0 + br, x0 - br, x0 - br], [y0 - br, y0 - br, y0 + br, y0 + br, y0 - br], 'white', linewidth=0.5)
                plt.annotate(r'0$\mu m$', xy=(x0 - br, y0 + br + 5), ha='center', color='white', va='top')
                plt.annotate(r'3$\mu m$', xy=(x0 + br, y0 + br + 5), ha='center', color='white', va='top')
                plt.annotate('%i' % p, xy=(x.iloc[0] - 2, y.iloc[0] - 2), xytext=(x.iloc[0] - 25, y.iloc[0] - 25), va='bottom', ha='right', arrowprops=dict(arrowstyle="-|>", facecolor='white', edgecolor='white'), color='white')

            # plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
            plt.imshow(self.getframe(0))

            # plot drift
            if len(drift) > 0:
                plt.plot(drift['x'], drift['y'], color='yellow', label='drift')
                t1 = tp.subtract_drift(t1, drift)

            plt.legend(loc=loc)
            plt.axis('off')
            plt.tight_layout()

            plt.savefig(self._fname[:-4] + '_trace.pdf', dpi=300)

            plt.show()

        return t1

    def kmean(self, frame=-1, n_cluster=5, show=False):
        """ show k-mean clustered image """
        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
        else:
            print('... {} is not in range'.format(frame))
            return False

        img = self.getframe(frame, types='uint8')
        img_flat = img.reshape((img.shape[0] * img.shape[1], 1))

        clt = MiniBatchKMeans(n_clusters=n_cluster)
        labels = clt.fit_predict(img_flat)
        quant = clt.cluster_centers_[labels]

        newimg = quant.reshape((img.shape))

        if show:
            plt.imshow(np.hstack((img, newimg)))
            plt.show()
            print(clt.cluster_centers_)

        return (pims.Frame(newimg), clt.cluster_centers_)

    def find_contour(self, frame=-1, threshold=None, show=True, min_size=30, max_size=-1, max_box=-1, plotmode='rbox'):
        """ find contour lines """
        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
        else:
            print('... {} is not in range'.format(frame))
            return False

        img = self.getframe(frame, types='uint8')

        # use threshold image
        if threshold is None:
            threshold = self.threshold(frame, show=False, erode_iter=2)

        # calculate contours
        im2, cnts, hi = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # sort by area
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        if max_box == -1:
            max_box = len(cnts)
        if max_size == -1:
            max_size = cv2.contourArea(cnts[0])

        # draw all detected contours
        c_img = img.copy()

        # prepare mask
        mask = np.zeros(img.shape, np.uint8)
        data = np.zeros((len(cnts), 15))
        good_idx = []

        if show:
            from matplotlib.patches import Ellipse
            fig = plt.figure(figsize=(10,5))
            ax = fig.gca()

        # check all contours
        for i, cnt in enumerate(cnts):
            M = cv2.moments(cnt)

            # filtering
            if M['m00'] < min_size:
                continue
            if M['m00'] > max_size:
                continue
            if i > max_box - 1:
                continue

            # find contour characteristics
            x, y, w, h = cv2.boundingRect(cnt)
            (rx, ry), (MA, ma), angleE = cv2.fitEllipse(cnt)
            (bx, by), (bw, bh), angleR = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(((bx, by), (bw, bh), angleR))
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
            perimeter = cv2.arcLength(cnt, True)

            white_color = np.random.random(3)/2 + 0.5
            if plotmode == 'box':
                # draw bounding box
                if show:
                    plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], color=white_color, alpha=0.8)
                angle = angleR
            elif plotmode == 'ellipse':
                # draw ellipse
                if show:
                    ells = Ellipse((rx, ry), MA, ma, angle=angleE, fill=False, linewidth=2, edgecolor=white_color, alpha=0.5)
                    ells.set_clip_box(ax.bbox)
                    ax.add_artist(ells)
                angle = angleE
                cx, cy = rx, ry
            else:
                if show:
                    box = np.array(box)
                    ax.plot(np.append(box[:,0], box[0,0]), np.append(box[:,1], box[0, 1]), color=white_color, alpha=0.8)
                angle = angleR
                cx, cy = bx, by

            if show:
                ax.annotate(str(i), xy=(x,y), color='white') #, fontsize='small')
                ax.annotate(str(i), xy=(x + img.shape[1],y), color='white') #, fontsize='small')

            data[i, :] = [frame, i, x, y, x+w, y+h, cx, cy, M['m00'], perimeter, MA, ma, angle, bw, bh]
            good_idx.append(i)

            # prepare total mask
            cv2.drawContours(mask, [cnt], 0, 255, -1)

        contour_data = pd.DataFrame(data[good_idx, :], index=range(len(good_idx)), columns=['frame', 'id', 'x0', 'y0', 'x1', 'y1', 'cx', 'cy', 'area', 'perimeter', 'major', 'minor', 'angle', 'width', 'height'])

        # calculate total c.o.m
        c_img[mask == 0] = 0
        M = cv2.moments(c_img)
        if M['m00'] == 0:
            cxy = 0, 0
        else:
            cxy = M['m10']/M['m00'], M['m01']/M['m00']

        # show result
        if show:
            ax.imshow(np.hstack((img, c_img)))
            ax.scatter([cxy[0] + img.shape[1]], [cxy[1]], s=160, c='white', marker='+')
            print("... total c.o.m: {:.2f}, {:.2f}".format(cxy[0], cxy[1]))

            plt.axis("image")
            plt.show()

        # prepare box array
        self._cnts = []
        self.box_arr_ = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_size:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            self.box_arr_.append([x, y, x+w, y+h])
            self._cnts.append(cnt)

        return contour_data

    # line profile
    def get_line_fit(self, loc=0, direction='x', nu=1000, show=True):
        """ get line profile and find peak """

        if len(self._wallinfo) > 1:
            si = self._wallinfo[0]
            fi = self._wallinfo[1]
        else:
            si = 0
            fi = self._width

        if direction == 'x':
            x = range(si, fi)
            y = self.mean()[loc, si:fi]
        else:
            x = range(0, self._height)
            y = self.mean()[:, loc]

        # keep previous fitting
        if len(self._kfit) == 0:
            self._kfit = [3.0, si + np.argmax(y), 2.0, self._baseline]

        if self._debug: print('... peak: %i' % (si + np.argmax(y)))
        kr = robust_gaussian_fit(x, y, nu=nu, initial=self._kfit, verb=self._debug)
        self._kfit = kr

        if show:
            msg = 'fit\nA: %.4g m: %.4g \ns: %.4g b: %.4g' % (kr[0], kr[1], kr[2], kr[3])
            yr = utils.gaussian(kr, x)
            plt.plot(x, y, label='raw')
            plt.plot(x, yr, '--', label=msg)
            plt.xlabel('x (pixels)')
            plt.ylabel('Intensity')
            plt.legend(loc='best')
            plt.savefig(self._fname[:-4] + '_rfit_l%i.pdf' % loc, dpi=200)
            print('... Amplitude: %f' % kr[0])
            print('... Mean: %f' % kr[1])
            print('... Deviation: %f' % kr[2])
            print('... Baseline: %f' % kr[3])
            plt.show()

        return kr

    def get_peaks(self, locs=-1, nu=1000, update=False, parallel=False):
        """ get peak position datasheet at locs """

        if locs == -1:
            locs = range(self._height)
        filename = '.' + self._fname.split('/')[-1] + '_peakinfo.csv'
        if (not update) and os.path.isfile(filename):
            if self._debug: print('... read from %s' % filename)
            self._peakdata = pandas.read_csv(filename)
        else:
            if self._debug: print('... create %s' % filename)
            d = np.zeros((self._height, 4)) - 1
            self._peakdata = pandas.DataFrame(d, index=range(self._height), columns=['loc', 'peak', 'delta', 'base'])

        count = 0
        if parallel:
            from multiprocessing import Pool
            get_line_fit = self.get_line_fit
            pool = Pool(processes=3)
            res = [pool.apply_async(get_line_fit, args=(x, 'x', 1000, False)) for x in locs]
            res = [p.get() for p in res]
            res = np.array(res, dtype=[('loc', 'i'), ('p', 'f'), ('d', 'f'), ('b', 'f')])
            res.sort(order='loc')

            self._peakdata['loc'] = res['loc']
            self._peakdata['peak'] = res['p']
            self._peakdata['delta'] = res['d']
            self._peakdata['base'] = res['b']
            count = len(locs)

        else:
            for i in tqdm.tqdm(locs):
                if update or (self._peakdata['loc'].loc[i] == -1):
                    kr = self.get_line_fit(i, direction='x', nu=nu, show=False)
                    self._peakdata['loc'].loc[i] = i
                    self._peakdata['peak'].loc[i] = kr[1]
                    self._peakdata['delta'].loc[i] = kr[2]
                    self._peakdata['base'].loc[i] = kr[3]
                    count += 1

        if count > 0:
            if self._debug: print('... save to %s' % filename)
            self._peakdata.to_csv(filename)

        return self._peakdata

    def show_peaks(self, update=False):
        """ show peakdata with image """

        if len(self._peakdata) == 0:
            self.get_peaks()

        ax = self._showimage(self.mean(), frameNumber=False)
        x = self._peakdata['loc']
        y = self._peakdata['peak']
        dy = self._peakdata['delta']
        ax.plot(y + 2.5 * dy, x, '.', color='gray', markersize=1, alpha=0.5, label='')
        ax.plot(y - 2.5 * dy, x, '.', color='gray', markersize=1, alpha=0.5, label=r'2.5$\sigma$')
        ax.plot(y, x, '.', markersize=1, label='peak', alpha=0.5)
        ax.legend(loc='best')
        plt.savefig(self._fname[:-4] + '_peak.pdf', dpi=300)
        plt.show()

    def show_angles(self, limits=-1, update=False):
        """ show peak positions and calculate angle """

        if limits == -1:
            limits = [0, self._height]

        if update and (len(self._peakdata) == 0):
            self.get_peaks()

        x = self._peakdata['loc'].loc[limits[0]:limits[1]]
        y = self._peakdata['peak'].loc[limits[0]:limits[1]]
        kr = robust_line_fit(x, y, nu=10.0, initial=[0.1, 0.0], verb=self._debug)
        yr = utils.line(kr, x)
        plt.plot(x, y)
        plt.plot(x, yr, '--', label='fit %.3g, %.3g' % (kr[0], kr[1]))
        plt.xlabel('locations [pixel]')
        plt.ylabel('peak positions [pixel]')

        msg = 'Shift: %.2g over %i     Angle: %.2g [deg]' % (yr.ptp(), limits[1] - limits[0], np.arctan(kr[0]) * 180.0 / np.pi)
        plt.annotate(msg, xy=(x.min(), y.min()), va='top', ha='left')
        plt.legend(loc='best')
        plt.savefig(self._fname[:-4] + '_angle.pdf', dpi=300)
        plt.show()

    def show_sigmas(self, limits=-1, update=False):
        """ show sigma and calculate diffusion coefficient """

        if limits == -1:
            limits = [0, self._height]

        if update and (len(self._peakdata) == 0):
            self.get_peaks()

        x = self._peakdata['loc'].loc[limits[0]:limits[1]].values
        y = self._peakdata['delta'].loc[limits[0]:limits[1]].values
        kr = robust_line_fit(x, y**2, nu=10.0, initial=[0.1, 0.0], verb=self._debug)
        yr = utils.line(kr, x)
        # plt.plot(x, y, label=r'$\sigma$')
        plt.plot(x, y**2, label=r'$\sigma^2$')
        plt.plot(x, yr, '--', label='fit %.3g, %.3g' % (kr[0], kr[1]))
        plt.xlabel('locations [pixel]')
        plt.ylabel('sigma [pixel]')

        msg = 'D/v: %.3g [pixel] Peclet Number: %.2g ' % (kr[0] / 2.0, 2.0 * 1 / kr[0])
        plt.annotate(msg, xy=(x.min(), (y.min())**2), va='top', ha='left')
        plt.legend(loc='best')

        sname = self._fname[:-4] + '_sigma.pdf'
        print('... save to %s' % sname)
        plt.savefig(sname, dpi=300)
        plt.show()

    def showline(self, lines=-1, dir='y', log=False, window=[0, 0], save=False, fit=False, smooth=False):
        """ plot line profiles """

        if lines == -1:
            if dir == 'y':
                lines = range(0, self._width, 50)
            else:
                lines = range(0, self._height, 50)

        tmp = self.mean()

        import matplotlib.ticker as ticker
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        ax1 = fig.add_subplot(121)
        self._im = ax1.imshow(tmp, clim=self._range, cmap=self._cmapname, origin='upper')
        ax1.axis('image')
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))
        ax2 = fig.add_subplot(122)
        if log:
            ax2.set_yscale('log')

        # print("... Locations ",lines)

        # set window
        if window[1] == 0:
            wl = 0
            if dir == 'x':
                wr = self._width
            else:
                wr = self._height
        else:
            wl = window[0]
            wr = window[1]
            if dir == 'x':
                ax1.hlines(window, 0, self._width, color='gray', alpha=0.5, linestyle='dashed')
            else:
                ax1.vlines(window, 0, self._height, color='gray', alpha=0.5, linestyle='dashed')

        print("... Window: %i, %i" % (wl, wr))
        res_line = np.zeros((len(lines), wr-wl))

        ax2.set_xlim(wl, wr)
        ax2.set_ylabel('Normalized Intensity')
        cmap = plt.get_cmap(self._cmapname)

        if dir == 'x':
            ax1.vlines(lines, wl, wr, color='yellow', alpha=0.5, linestyle='dashed')
            ax2.set_xlabel('y (pixel)')
            i = 0
            for l in lines:
                color = cmap(float(i) / len(lines))
                if smooth:
                    ax2.plot(range(wl, wr + 1), utils._smooth(tmp[wl:wr, l]), label=l, color=color)
                    res_line[i, :] = utils._smooth(tmp[wl:wr, l])[:-1]
                else:
                    ax2.plot(range(wl, wr), tmp[wl:wr, l], label=l, color=color)
                    res_line[i, :] = tmp[wl:wr, l]
                ax1.annotate(l, xy=(l, wl), va='bottom', color='white')
                print("... Loc: %i Peak: %i Value: %f" % (l, np.argmax(tmp[wl:wr, l]), np.max(tmp[wl:wr, l])))
                if fit:
                    out = fitGaussian(np.range(wl, wr), tmp[wl:wr, l], baseline=self._baseline)
                    ax2.plot(range(wl, wr), out.best_fit + self._baseline, '--', color=color)
                    msg = "%.2f,%.2f" % (out.best_values['center'], out.best_values['sigma'])
                    fx = np.argmax(tmp[:, l])
                    fy = np.max(tmp[:, l]) + self._baseline
                    ax2.annotate(msg, xy=(fx, fy), va='baseline', ha=self._beamdirection)
                    self._beaminfo.append([l, out.best_values['center'], out.best_values['sigma']**2])
                i += 1
        elif dir == 'y':
            ax1.hlines(lines, wl, wr, color='yellow', alpha=0.5, linestyle='dashed')
            ax2.set_xlabel('x (pixel)')
            i = 0
            for l in lines:
                color = cmap(float(i) / len(lines))
                if smooth:
                    ax2.plot(range(wl, wr + 1), utils._smooth(tmp[l, wl:wr]), label=l, color=color)
                    res_line[i, :] = utils._smooth(tmp[l, wl:wr])[:-1]
                else:
                    ax2.plot(range(wl, wr), tmp[l, wl:wr], label=l, color=color)
                    res_line[i, :] = tmp[l, wl:wr]
                ax1.annotate(l, xy=(wl, l), ha='right', color='white')
                print("... Loc: %i Peak: %i Value: %f" % (l, np.argmax(tmp[l, wl:wr]), np.max(tmp[l, wl:wr])))
                if fit:
                    out = fitGaussian(np.arange(wl, wr), tmp[l, wl:wr], baseline=self._baseline)
                    ax2.plot(range(wl, wr), out.best_fit + self._baseline, '--', color=color)
                    msg = "%.2f,%.2f" % (out.best_values['center'], out.best_values['sigma'])
                    fx = np.argmax(tmp[l, :])
                    fy = np.max(tmp[l, :]) + self._baseline
                    ax2.annotate(msg, xy=(fx, fy), va='baseline', ha=self._beamdirection)
                    self._beaminfo.append([l, out.best_values['center'], out.best_values['sigma']**2])
                i += 1

        else:
            return False

        plt.tight_layout()
        plt.legend(loc='best')

        if save:
            linestr = "_" + dir + "l".join(['_' + str(l) for l in lines])
            filename = self._fname[:-4] + linestr + '.pdf'
            plt.savefig(filename, dpi=100)
        plt.show()

        return res_line

    def lineprofile(self, lpos, frame=-1, peaks=[], amps=[], sigmas=[], baseline=0.0, single=True, lmax=-1, shift=0):
        """ plot line profile """

        if frame == -1:
            im = self.mean()
        else:
            im = self.getframe(frame)

        Imax = np.max(im)
        Imin = np.min(im)
        y = (im[lpos, :] - Imin) / (Imax - Imin)
        x = np.arange(len(y))
        # plt.plot(x, y, '.', label='raw', markersize=3, alpha=0.8)
        plt.plot(x + shift, y, '-', color='gray', label='raw', alpha=0.5)
        plt.xlabel('position [pixel]')
        plt.ylabel('Relative Intensity')

        # single peak
        if len(peaks) == 1:
            if len(amps) == 0: amps = [1.0]
            if len(sigmas) == 0: sigmas = [1.0]
            kr = robust_gaussian_fit(x, y, nu=30.0, initial=[amps[0], peaks[0], sigmas[0], baseline], verb=self._debug)
            yr = gaussian(kr, x)
            plt.plot(x + shift, yr, '-', label='peak: %.1f, %.1f' % (kr[1], kr[2]))
            print('A: %.4f mu: %.4f sigma: %.4f base: %.4f' % (kr[0], kr[1], kr[2], kr[3]))
        # double peaks
        elif len(peaks) == 2:
            if len(amps) == 0: amps = [1.0, 1.0]
            if len(sigmas) == 0: sigmas = [1.0, 1.0]
            kr = robust_gaussian2_fit(x, y, nu=30.0, initial=[peaks[0], peaks[1], sigmas[0], sigmas[1], amps[0], amps[1], baseline], verb=self._debug)

            if single:
                yr1 = gaussian([kr[4], kr[0], kr[2], kr[6]], x)
                yr2 = gaussian([kr[5], kr[1], kr[3], kr[6]], x)
                plt.plot(x[:int(kr[0] + 3.5 * kr[2])] + shift, yr1[:int(kr[0] + 3.5 * kr[2])], '-', label='peak1: %.1f, %.1f' % (kr[0], kr[2]))
                plt.plot(x[int(kr[1] - 3.5 * kr[3]):] + shift, yr2[int(kr[1] - 3.5 * kr[3]):], '-', label='peak2: %.1f, %.1f' % (kr[1], kr[3]))
            else:
                yr = gaussian2(kr, x)
                plt.plot(x + shift, yr, '--')

            print('A: %.4f mu: %.4f sigma: %.4f base: %.1f' % (kr[4], kr[0], kr[2], kr[6]))
            print('A: %.4f mu: %.4f sigma: %.4f base: %.1f' % (kr[5], kr[1], kr[3], kr[6]))
        # triple peaks
        elif len(peaks) == 3:
            if len(amps) == 0: amps = [1.0, 1.0, 1.0]
            if len(sigmas) == 0: sigmas = [1.0, 1.0, 1.0]
            kr = robust_gaussian3_fit(x, y, nu=30.0, initial=[amps[0], peaks[0], sigmas[0], amps[1], peaks[1], sigmas[1], amps[2], peaks[2], sigmas[2], baseline])

            if single:
                yr1 = gaussian([kr[0], kr[1], kr[2], kr[9]], x)
                yr2 = gaussian([kr[3], kr[4], kr[5], kr[9]], x)
                yr3 = gaussian([kr[6], kr[7], kr[8], kr[9]], x)
                plt.plot(x[:int(kr[1] + 3.5 * kr[2])] + shift, yr1[:int(kr[1] + 3.5 * kr[2])], '-', label='peak1: %.1f, %.1f' % (kr[1], kr[2]))
                plt.plot(x[int(kr[4] - 3.5 * kr[5]):int(kr[4] + 3.5 * kr[5])] + shift, yr2[int(kr[4] - 3.5 * kr[5]):int(kr[4] + 3.5 * kr[5])], '-', label='peak2: %.1f, %.1f' % (kr[4], kr[5]))
                plt.plot(x[int(kr[7] - 3.5 * kr[8]):] + shift, yr3[int(kr[7] - 3.5 * kr[8]):], '-', label='peak3: %.1f, %.1f' % (kr[7], kr[8]))
            else:
                yr = gaussian3(kr, x)
                plt.plot(x + shift, yr, '--')

            print('A: %.4f mu: %.4f sigma: %.4f base: %.4f' % (kr[0], kr[1], kr[2], kr[9]))
            print('A: %.4f mu: %.4f sigma: %.4f base: %.4f' % (kr[3], kr[4], kr[5], kr[9]))
            print('A: %.4f mu: %.4f sigma: %.4f base: %.4f' % (kr[6], kr[7], kr[8], kr[9]))

        plt.legend(loc='best')
        if lmax == -1: lmax = np.max(x)
        plt.xlim(0, lmax)
        plt.savefig('%s-l%i.pdf' % (self._fname, lpos), dpi=150)
        plt.show()

    def plotYaxis(self, bg=0.0, bgstd=0.0, window=None):
        utils._plotAxis(self.mean(), 0, background=bg, backstd=bgstd, window=window)

    def plotXaxis(self, bg=0.0, bgstd=0.0, window=None):
        utils._plotAxis(self.mean(), 1, background=bg, backstd=bgstd, window=window)

    # plot functions
    def _showimage(self, img, simple=False, frameNumber=True):
        """ base plot function with axis  and colorbar """

        plt.clf()
        self._im = plt.imshow(img, clim=self._range, cmap=self._cmapname, origin='image')
        ax = self._im.axes

        if simple:
            plt.axis('off')
        else:
            if frameNumber and (self._frameN > 1):
                plt.annotate('%d/%d' % (self._curframe, self._frameN), xy=(0, 0))
            elif (self._frameN > 1):
                plt.annotate('%d' % (self._frameN), xy=(0, 0))

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            plt.axis('on')
            ax.axis((0, img.shape[1], img.shape[0], 0))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            plt.colorbar(self._im, cax=cax)
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')

        return ax

    def showframe(self, frame=-1, simple=False):
        """ plot frame with axis """

        # fig.canvas.mpl_connect('key_press_event', self.on_key_event)
        if frame == -1:
            frame = self._curframe
        if -1 < frame < self._frameN:
            img_f = self.getframe(frame, types='float')
            self._curframe = frame
            self._showimage(img_f, simple=simple)
            plt.show()

    def shownext(self, simple=False):
        self._curframe += 1
        if self._curframe == self._frameN:
            self._curframe = 0

        if self._im is None:
            self._showimage(self.getframe(self._curframe), simple=simple)
            plt.show()
        else:
            self._im.set_data(self.getframe(self._curframe))
        # print('... frame position %i out of %i' % (self._curframe, self._frameN))

    def showprevious(self, simple=False):
        self._curframe -= 1
        if self._curframe == -1:
            self._curframe = self._frameN - 1

        if self._im is None:
            self._showimage(self.getframe(self._curframe), simple=simple)
            plt.show()
        else:
            self._im.set_data(self.getframe(self._curframe))
        # print('... frame position %i out of %i' % (self._curframe, self._frameN))

    def showMean(self, simple=False):
        ax = self._showimage(self.mean(), simple=simple, frameNumber=False)
        self._im = ax

    def showMax(self, view=True, simple=False):
        ax = self._showimage(self.max(), simple=simple, frameNumber=False)
        self._im = ax

    def showMin(self, view=True, simple=False):
        ax = self._showimage(self.min(), simple=simple, frameNumber=False)
        self._im = ax

    def hist(self, frame=-1, img=None):
        """ plot histogram """

        if frame == -1:
            frame = self._curframe
        if frame in self._frames:
            self._curframe = frame
        else:
            print('... {} is not in range'.format(frame))
            return False
        if img is None:
            img = self.getframe(frame, types='uint8')

        h = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(h)
        plt.xlim(0, 256)
        plt.ylim(h.min(), h.max())
        plt.show()

    def box_view(self, box_number=0):
        """ show box area """
        if len(self.box_arr_) == 0:
            print("... use find_contour first")
            return False

        if box_number in range(len(self.box_arr_)):
            x0, y0, x1, y1 = self.box_arr_[box_number]
            if len(self._newimages) > 0:
                return pims.Frame(self._newimages[:, y0:y1, x0:x1])
            else:
                imgs = np.array(self._images[:])
                return pims.Frame(imgs[:, y0:y1, x0:x1])

    # save
    def save_tif(self, appendix=-1, box_arr=None, margin=0):
        """ save as tif format for saving data """

        if appendix == -1:
            appendix = ''
        if len(self._nd2) > 0:
            nd_code = '-m%ic%i-' % (self._nd2_m, self._nd2_c)
        else:
            nd_code = ''

        # check box information
        if len(box_arr) == 0:
            img_array = np.array(self._images[:])
            savefname = self._fname[:-4] + nd_code + appendix + '.tif'

            if os.path.exists(savefname):
                a = input('... overwrite %s' % savefname)
                if a.upper() in ['NO', 'N']:
                    return False

            print('... save to %s' % savefname)
            tifffile.imsave(savefname, img_array)

        # with box selections
        elif len(box_arr) > 0:
            img_array = np.array(self._images[:])

            # for each boxes
            for (i, box) in enumerate(box_arr):
                savefname = self._fname[:-4] + nd_code + 'crop_%i' % i + '.tif'

                if os.path.exists(savefname):
                    a = input('... overwrite %s' % savefname)
                    if a.upper() in ['NO', 'N']:
                        continue

                [x0, y0, x1, y1] = np.array(box) + np.array([-margin, -margin, margin, margin])
                # adjust values not to overbound
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(self._height, x1), min(self._width, y1)

                print('... save to %s' % savefname)
                tifffile.imsave(savefname, img_array[:, y0:y1 + 1, x0:x1 + 1])

    def saveframe(self, filename, simple=False):
        """ save one frame as plot format """
        self._showimage(self.getframe(self._curframe), simple=simple)
        plt.savefig(filename, dpi=100)

    def save_movie(self, zoomfactor=1.0, savename='None', update=False, verbose=True):
        """
        save tif file as mp4 using moviepy library. The duration of movie file
        is determined as 3 times of realtime video.

        Input:
        zoomfactor = 0.5 (default), save image using resampling
        savename = default format [tif file name]_z[zoomefactor].mp4
        show = show movie file in jupyter (not implemented)

        Return:
        VidoeClip object in moviepy library
        """

        if (savename == 'None'):
            savename = '%s_z%.1f.mp4' % (self._fname[:-4], zoomfactor)
            if not update:
                if os.path.exists(savename):
                    if verbose:
                        print('... movie file already exists: %s' % savename)
                        print('... if you want to play in jupyter, ')
                        print('...  type <tif object>._animation.ipython_display(fps=20, autoplay=True)')
                        print('...  type <tif folder object>.get_tif()._animation.ipython_display(fps=20, autoplay=True)')
                    return 0

        if self._frameN == 1:
            if verbose: print('... not movie file')
            return 0

        from moviepy.editor import VideoClip

        cmap = plt.get_cmap(self._cmapname)

        def make_frame(t):
            self._curframe = int(t * (self._frameN - 1) / (self._duration * 3.0))
            img0 = self.getframe(self._curframe, types='uint8')
            if zoomfactor != 1.0:
                img0 = cv2.resize(img0, None, fx=zoomfactor, fy=zoomfactor, interpolation=cv2.INTER_CUBIC)
            img = np.delete(cmap(img0), 3, 2)
            return (img * 255.0).astype('uint8')
            #return img.astype('uint8')

        animation = VideoClip(make_frame, duration=self._duration * 3.0)

        #if savename == 'None':
        #    savename = self._fname[:-4] + '_z' + str(zoomfactor) + '.avi'
        animation.write_videofile(savename, fps=20, codec='mpeg4', threads=2)
        self._animation = animation
        if verbose:
            print('... if you play in jupyter, ')
            print('...  type <tif object>._animation.ipython_display(fps=20, autoplay=True)')
            print('...  type <tif folder object>.get_tif()._animation.ipython_display(fps=20, autoplay=True)')

    # experimental data
    def beamInfo(self, showGraph=False):
        """ show beam information from peak positions """

        if len(self._beaminfo) <= 0:
            print('...no beam info - run showline command')
            return
        print('location /   peak   /   sigma^2')
        print('--------------------------------')
        x = []
        y = []
        z = []
        for i in range(len(self._beaminfo)):
            print(self._beaminfo[i])
            x.append(self._beaminfo[i][0])
            y.append(self._beaminfo[i][1])
            z.append(self._beaminfo[i][2])
        mod = LinearModel()

        pars = mod.guess(y, x=x)
        out1 = mod.fit(y, pars, x=x)
        msg1 = 'Slope: %.2f\nAngle: %.2f [deg]' % \
            (out1.best_values['slope'], np.arctan(out1.best_values['slope']) * 180.0 / np.pi)
        print(msg1)

        pars = mod.guess(z, x=x)
        out2 = mod.fit(z, pars, x=x)
        msg2 = 'Slope: %.2f\nD: %.2f [pixel^2/frame]\nvel: %.2f [pixel/frame]' % \
            (out2.best_values['slope'],
             out2.best_values['slope'] * self._beamvel / 2.0,
             self._beamvel)
        print(msg2)

        if showGraph:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121)
            ax1.plot(x, y, 'o', label=msg1)
            ax1.plot(x, out1.best_fit, '--')
            plt.xlabel('l [pixels]')
            plt.ylabel('peak [pixels]')
            plt.legend(loc='best')

            ax2 = fig.add_subplot(122)
            ax2.plot(x, z, 'o', label=msg2)
            ax2.plot(x, out2.best_fit, '--')
            plt.xlabel('l [pixels]')
            plt.ylabel(r'$\sigma^2$ [pixel^2]')
            plt.legend(loc='best')

            filename = self._fname[-3] + '_beaminfo.pdf'
            plt.savefig(filename, dpi=100)
            plt.show()

    def channelInfo(self, window=11, compareRatio=0.2, minimumIntensity=0.1, show=True):
        """ find channel wall information """
        result = utils._find_channel(self._imgMean, window=window,
                                     compareRatio=compareRatio, minimumIntensity=minimumIntensity)

        if show:
            if len(result) == 4:
                ax = self._showimage(self.mean(), simple=False)
                ax.vlines([result[0], result[1]], 0, self._height, color='w')
                ax.hlines([result[2], result[3]], result[0], result[1], color='w')
            elif len(result) == 3:
                ax = self._showimage(self.mean(), simple=False)
                ax.vlines([result[0], result[1]], 0, self._height, color='w')
                ax.hlines(result[2], result[0], result[1], color='w')
            elif len(result) == 2:
                ax = self._showimage(self.mean(), simple=False)
                ax.vlines([result[0], result[1]], 0, self._height, color='w')
            plt.show()

        self._wallinfo = result

        return result

    def check_frame_angle(self, minthresh=60, maxthresh=110, minlength=250):
        """ check rotation of image - detect edges and lines and calculate angle """

        img = np.uint8((self._imgMean - self._imgMean.min()) / (self._imgMean.max() - self._imgMean.min()) * 255)
        edges = cv2.Canny(img, minthresh, maxthresh)
        lines = cv2.HoughLines(edges, 1, np.pi / 720.0, minlength)

        plt.imshow(edges)
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
            plt.plot((x1, x2), (y1, y2))
            print('... theta: %.4f [deg]  (x0, y0) : (%i, %i)' % (theta * 180.0 / np.pi, x0, y0))

        plt.xlim([0, self._width])
        plt.ylim([self._height, 0])
        plt.show()

    def calAreabyIntensity(self, ranges, frame=-1, show=True):
        if frame == -1:
            frame = self._curframe
        else:
            self._curframe = frame
        img = self.getframe(frame)
        mask1 = img < ranges[0]
        mask2 = img > ranges[1]
        mask = mask1 | mask2
        if show:
            plt.imshow(mask, cmap=plt.cm.gray)
            plt.axis('off')
            plt.show()

        x1, y1 = img.shape
        size = x1 * y1 - np.sum(mask)
        print('... size: %d [pixel^2]' % size)
        intensity = np.sum(img[~mask])
        print('... intensity sum: %.3g' % intensity)


def fitGaussian(x, data, baseline=0.0, verb=False):
    """ use gaussion model to fit result """

    tmp = np.copy(data)
    tmp -= baseline
    peakx = x[np.argmax(data)]
    tmp[np.where(tmp < 0)] = 0.0

    mod = GaussianModel(missing='drop', independent_vars=['x'])

    # pars = mod.guess(data, x=x)
    out = mod.fit(tmp, center=peakx, amplitude=1.0, sigma=1.0, x=x)

    if verb:
        print(out.fit_report(min_correl=0.25))

    return out


def showmultiple(imgs):
    """ show multiple images """

    totalx, maxy, startx = 0, 0, 0

    for img in imgs:
        totalx += img.shape[1]
        maxy = max(maxy, img.shape[0])

    res = np.zeros((maxy, totalx))
    for img in imgs:
        img_n = img / np.max(img)
        res[0:img.shape[0], startx:startx+img.shape[1]] = img_n
        startx += img.shape[1]

    plt.imshow(res)
    plt.show()

# vim:foldmethod=indent:foldlevel=0
