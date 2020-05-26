#!/usr/bin/env python
"""
base_filters.py - microscopy images class

"""

import sys
import pims
from tqdm import tnrange
import time

import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from functools import wraps

from .base import ImageBase
from .base import _rescale_to_dtype

__author__ = 'Sungcheol Kim <kimsung@us.ibm.com>'
__version__ = '1.0.0'

__all__ = ('ImageFilter', 'NoiseGater', '_operation_list')

image_operations = [('exposure', ['contrast', 'equalization', 'adapt', 'local', 'gamma', 'sigmoid']),
                    ('denoise', ['rof', 'tvl', 'wavelet', 'bilateral2', 'deforest', 'bilateral', 'gaussian', 'median', 'fastNl']),
                    ('filters', ['sharpen', 'median3d', 'min3d', 'mean3d', 'scharr', 'prewitt', 'sobel', 'removecell', 'canny']),
                    ('threshold', ['otsu', 'otsu2', 'adaptive', 'kmean', 'triangle', 'sauvola', 'yen'])]


def _operation_list(name):
    for (ops, methods) in image_operations:
        if name == ops:
            return name + ": " + ', '.join(methods)


class ImageFilter(ImageBase):

    def __init__(self, objects, method='tifffile', debug=False, **kwargs):
        super().__init__(objects, method=method, debug=debug, **kwargs)

        self._filter_history = []
        self._filter_history_number = np.zeros(self._meta.N(), dtype='uint8')
        self._threshold_images = []

    def log_filter(func):
        """ record filter history """
        @wraps(func)
        def preprocess(self, *args, **kwargs):
            i = kwargs.get('frame', -1)
            img = func(self, *args, **kwargs)
            if i == -1: i = self._curframe
            self._filter_history_number[i] += 1
            self._filter_history.append((i, img))
            if self._single:
                self._images = img
            else:
                self._images[i, :, : ] = img
            if self._debug: 
                print('... register [{}][{}]: {:.4f}, {:.4f}, {:.4f}, {}'.format(i, 
                    self._filter_history_number[i], img.min(), img.mean(), img.max(), img.dtype))
            return img
        return preprocess

    def process_all_from_string(self, instr_str):
        keep = self._debug
        self._debug = False
        for i in tnrange(self._frames):
            self._process_from_string(i, instr_str)
        self._debug = keep

    def _process_from_string(self, frame, instr_str):
        """ apply filters from instruction string
        Example:
        --------------------------------
        ins_str = \"""
        exposure, gamma, intensity=0.3
        denoise, gaussian
        \"""
        --------------------------------
        """
        lines = instr_str.splitlines()

        ops_list = [i for (i, _) in image_operations]

        for i, line in enumerate(lines):
            if len(line) == 0: continue

            components = line.split(',')
            method, intensity, kwargs = "", "", ""

            if components[0].strip() in ops_list:
                ops = components[0]

                try:
                    method = components[1].strip()
                    method = 'method="{}"'.format(method)
                except:
                    method = ""

                try:
                    index = components[2].strip().find("intensity")
                    if index > -1:
                        intensity = ", " + components[2].strip()
                    else:
                        intensity = ""
                        kwargs = ", " + components[2].strip()
                except:
                    intensity = ""

                try:
                    kwargs = ", " + components[3].strip()
                except:
                    if len(kwargs) == 0:
                        kwargs = ""

                commands = 'self.{}({}, {}{}{})'.format(ops, frame, method, intensity, kwargs)
                if self._debug: print('[{}] {}'.format(i, commands))
                eval(commands)

    # image processing
    @log_filter
    def exposure(self, frame=-1, method='adapt', intensity=-1):
        return _exposure(self.getframe(frame=frame, types='float'), method=method, debug=self._debug, intensity=intensity)

    @log_filter
    def denoise(self, frame=-1, method='fastNl', intensity=-1, **kwargs):
        return _denoise(self.getframe(frame=frame, types='orig'), method=method, debug=self._debug, intensity=intensity, **kwargs)

    @log_filter
    def filters(self, frame=-1, method='sharpen', intensity=-1, **kwargs):
        return self._filters(self.getframe(frame=frame, types='orig'), method=method, debug=self._debug, intensity=intensity, **kwargs)

    def _filters(self, img, method='', intensity=-1, debug=False, **kwargs):
        """ apply other filters """
        if 'cv2' not in dir(): import cv2

        if method == 'sharpen':
            if debug: print('... sharpen')
            if intensity == -1: intensity = 0.3
            elif intensity == 'full': intensity = 0.5
            img = _rescale_to_dtype(img, 'float')
            blur = kwargs.pop('blur', 3)
            img_g = cv2.GaussianBlur(img, (blur, blur), 0)
            U = img + (img - img_g) * intensity

        elif method == 'median3d':
            if debug: print('... background subtraction by median z-projection')
            if intensity == -1: intensity = self.tmedian()
            U = _rescale_to_dtype(img - intensity, 'float')

        elif method == 'min3d':
            if debug: print('... background subtraction by median z-projection')
            if intensity == -1: intensity = self.tmin()
            U = _rescale_to_dtype(img - intensity, 'float')

        elif method == 'mean3d':
            if debug: print('... background subtraction by median z-projection')
            if intensity == -1: intensity = self.tmean()
            U = _rescale_to_dtype(img - intensity, 'float')

        elif method == 'sobel':
            if debug: print('... edge detection using sobel filter')
            from skimage.filters import sobel
            U = sobel(img)

        elif method == 'canny':
            if debug: print('... edge detection using canny filter')
            img = _rescale_to_dtype(img, 'uint8')
            U = cv2.Canny(img, 50, 200, None, 3)

        elif method == 'prewitt':
            if debug: print('... edge detection using prewitt filter')
            from skimage.filters import prewitt
            U = prewitt(img)

        elif method == 'roberts':
            if debug: print('... edge detection using roberts filter')
            from skimage.filters import roberts
            U = roberts(img)

        elif method == 'scharr':
            if debug: print('... edge detection using scharr filter')
            from skimage.filters import scharr
            U = scharr(img)

        elif method == 'laplace':
            if debug: print('... edge detection using laplace filter')
            from skimage.filters import laplace
            ksize = kwargs.pop('ksize', 3)
            U = laplace(img, ksize=ksize)

        elif method == 'removecell':
            if debug: print('... cell remover using kmean filter')
            if intensity == -1: intensity = 0.3
            U = self._remove_cell(img, percentile=intensity, **kwargs)

        else:
            print(_operation_list("filters"))
            U = img

        return pims.Frame(U)

    @log_filter
    def threshold(self, frame=-1, method='otsu', erode_iter=0, **kwargs):
        return _threshold(self.getframe(frame=frame, types='orig'), method=method, erode_iter=erode_iter, debug=self._debug, **kwargs)

    def kmean(self, frame=-1, n_cluster=5, show=False):
        return _kmean(self.getframe(frame=frame), n_cluster=n_cluster, show=show)



def _exposure(img, method='adapt', intensity=-1, debug=False):
    """ preprocess image mostly by renormalization method: contrast, equalization, adapt, gamma, sigmoid """
    import skimage.exposure

    # histogram filter
    if method == 'contrast':
        if debug: print('... contrast histogram')
        if 0 < intensity < 49:
            p2_f, p98_f = intensity, 100 - intensity
        else:
            p2_f, p98_f = 2, 98
        p2, p98 = np.percentile(img, (p2_f, p98_f))
        U = skimage.exposure.rescale_intensity(img, in_range=(p2, p98))

    elif method == 'equalization':
        if debug: print('... global equalize')
        U = skimage.exposure.equalize_hist(img)

    elif method == 'adapt':
        if debug: print('... contrast limited adaptive histogram equalization (CLAHE)')
        if intensity == -1: intensity = int(img.shape[0]/8)
        U = skimage.exposure.equalize_adapthist(_rescale_to_dtype(img, 'uint16'), kernel_size=intensity, clip_limit=0.03)

    elif method == 'local':
        if debug: print('... local histogram equalization')
        if intensity == -1: intensity = 30
        from skimage.morphology import disk
        from skimage.filters import rank
        selem = disk(intensity)
        U = rank.equalize(_rescale_to_dtype(img, 'uint8'), selem=selem)

    elif method == 'gamma':
        if debug: print('... gamma equalize')
        if intensity == -1: intensity = 0.80
        U = skimage.exposure.adjust_gamma(img, gamma=intensity)

    elif method == 'sigmoid':
        if debug: print('... sigmoid equalize')
        if intensity == -1: intensity = 0.5
        U = skimage.exposure.adjust_sigmoid(img, cutoff=intensity)

    else:
        print(_operation_list("exposure"))
        U = img

    if skimage.exposure.is_low_contrast(U):
        print('... low contrast image')

    return pims.Frame(U)


def _denoise(img, method='fastNl', intensity=-1, debug=False, **kwargs):
    """ denoise frame """
    from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle, denoise_wavelet, denoise_bilateral
    if 'cv2' not in dir(): import cv2

    if method in ['bilateral', 'fastNl']:
        img = _rescale_to_dtype(img, 'uint8')
    elif method == 'deforest':
        img = _rescale_to_dtype(img, 'float')

    if method == 'rof':
        if debug: print('... total-variation denoising using split-Bregman optimization')
        if intensity == -1: intensity = 70
        elif intensity == 'full': intensity = 10
        U = denoise_tv_bregman(img, intensity, **kwargs)

    elif method == 'tvl':
        if debug: print('... total-variation denoising on n-dimensional images')
        if intensity == -1: intensity = 0.01
        elif intensity == 'full': intensity = 0.05
        U = denoise_tv_chambolle(img, intensity, **kwargs)

    elif method == 'wavelet':
        if debug: print('... perform wavelet denoising')
        if intensity == -1: intensity = 1
        elif intensity == 'full': intensity = 3
        U = denoise_wavelet(img, wavelet_levels=intensity, **kwargs)

    elif method == 'bilateral2':
        if debug: print('... denoise image using bilateral filter')
        if intensity == -1: intensity = 1
        elif intensity == 'full': intensity = 3
        U = denoise_bilateral(img, sigma_spatial=intensity, multichannel=False, **kwargs)

    elif method == 'deforest':
        if intensity == -1: intensity = 3.0
        elif intensity == 'full': intensity = 3.0
        ng0 = NoiseGater(img, gamma=intensity, debug=debug, **kwargs)
        U = ng0.clean()

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

    elif method == 'fastNl':
        if intensity == -1: intensity = 5
        elif intensity == 'full': intensity = 9
        U = cv2.fastNlMeansDenoising(img, h=intensity, **kwargs)

    else:
        print(_operation_list("denoise"))
        U = img

    return pims.Frame(U)


def _threshold(img, method='otsu', erode_iter=0, show=True, **kwargs):
    """ automatic threshold finder """

    if 'cv2' not in dir(): import cv2

    # find threshold
    if method == 'otsu':
        img = _rescale_to_dtype(img, 'uint8')
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if debug: print('... threshold operation using otsu algorithm: {}'.format(ret))

    elif method == 'otsu2':
        from skimage.filters import threshold_otsu
        ret = threshold_otsu(img)
        th = np.zeros_like(img, dtype='uint8')
        th[img > ret] = 255
        if debug: print('... threshold operation using otsu2 algorithm: {}'.format(ret))

    elif method == 'adaptive':   # not working well
        if debug: print('... threshold operation using adaptive algorithm')
        img = _rescale_to_dtype(img, 'uint8')
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ret = 0

    elif method == 'kmean':
        kth, centers = _kmean(img, n_cluster=2, show=False)
        ret = np.mean(centers)
        th = np.zeros_like(kth, dtype='uint8')
        th[kth > ret] = 255
        if debug: print('... threshold operation using kmean algorithm: {}, {}'.format(centers[0], centers[1]))

    elif method == 'triangle':
        from skimage.filters import threshold_triangle
        ret = threshold_triangle(img)
        th = np.zeros_like(img, dtype='uint8')
        th[img > ret] = 255
        if debug: print('... threshold operation using triangle algorithm: {}'.format(ret))

    elif method == 'yen':
        from skimage.filters import threshold_yen
        ret = threshold_yen(img)
        th = np.zeros_like(img, dtype='uint8')
        th[img > ret] = 255
        if debug: print('... threshold operation using yen algorithm: {}'.format(ret))

    elif method == 'sauvola':
        from skimage.filters import threshold_sauvola
        wsize = kwargs.pop('window_size', 15)
        ret = threshold_sauvola(img, window_size=wsize)
        th = np.zeros_like(img, dtype='uint8')
        th[img > ret] = 255
        ret = ret.mean()
        if debug: print('... threshold operation using sauvola algorithm: {}'.format(ret))

    else:
        print(_operation_list("threshold"))
        raise TypeError('... no threshold method: {}'.format(method))

    # morphological transformation
    if erode_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.erode(th, kernel, iterations=erode_iter)
        th = cv2.dilate(th, kernel, iterations=erode_iter)

    if show:
        M = cv2.moments(th)
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        area = M['m00']

        plt.figure(figsize=(10, 5))
        plt.imshow(np.hstack((_rescale_to_dtype(img, 'uint8'), th)))
        plt.axis('off')
        plt.show()
        print('...thresh: %i, Area: %i' % (ret, area))
        print('...center: (%i, %i)' % (cx, cy))

    return th


def _kmean(img, n_cluster=5, show=False):
    """ show k-mean clustered image """
    img_flat = img.reshape((img.shape[0] * img.shape[1], 1))

    from sklearn.cluster import MiniBatchKMeans

    clt = MiniBatchKMeans(n_clusters=n_cluster)
    labels = clt.fit_predict(img_flat)
    quant = clt.cluster_centers_[labels]

    newimg = quant.reshape((img.shape))

    if show:
        plt.imshow(np.hstack((img, newimg)))
        plt.show()
        print(clt.cluster_centers_)

    return (pims.Frame(newimg), clt.cluster_centers_)


def _remove_cell(img, percentile=0.1, n_cluster=2, debug=False):
    """ find condensate using double threshold """

    # use kmean to select foreground and background
    th, center = _kmean(img, n_cluster=n_cluster, show=debug)

    # prepare divider value between foreground and max intensity
    background, foreground = np.min(center), np.max(center)
    divider = foreground + (img.max() - foreground) * percentile

    U2 = img.copy()
    U2[U2 < divider] = divider

    if debug:
        print('thresh1: %i, Area sum: %i' % (background, th.sum() / 255.0))

    return U2


@njit(fastmath=True, cache=True)
def _build_hanning_window(Nx, Ny):
    """ prepare hanning window """
    hanning_window_2D = lambda x, y: np.power(np.sin((x + 0.5)*np.pi / Nx), 2.0) * np.power(np.sin((y + 0.5) * np.pi / Ny), 2.0)
    hanning = np.zeros((Nx, Ny))

    for x in range(Nx):
        for y in range(Ny):
            hanning[x, y] = hanning_window_2D(x, y)

    return hanning


class NoiseGater:
    """ decrease shot noise using Deforest (2017) paper """

    def __init__(self, img, width=8, step=4, gamma=3.0, beta_percentile=50, beta_count=0, types='float', mode='numpy', debug=False):

        # tried to use pyfftw, but it was slower than numpy fft
        import pyfftw
        import cv2

        # prepare variables
        extended_img = cv2.copyMakeBorder(img, 2*width, 2*width, 2*width, 2*width, cv2.BORDER_REPLICATE)
        self._img = extended_img
        self._xwidth, self._ywidth = width, width
        self._xstep, self._ystep = step, step

        self._types = types
        self._debug = debug
        #self._nthread = multiprocessing.cpu_count()
        self._nthread = 1
        self.NX, self.NY = 2 * self._xwidth + 1, 2 * self._ywidth + 1

        self.image_section = pyfftw.zeros_aligned((self.NX, self.NY), dtype='float32')
        self.fourier_section = pyfftw.zeros_aligned((self.NX, self.NY), dtype='complex64')

        self.method = mode
        self.gamma = gamma

        self.coords = self._build_coordinates()
        self.coord_N = len(self.coords)

        self.hanning = _build_hanning_window(self.NX, self.NY)

        if beta_count == 0:
            beta_count = int(self.coord_N * 1.0)
        self.beta_count = beta_count
        self.beta_percentile = beta_percentile
        self.beta = []

        pyfftw.forget_wisdom()

    def _build_coordinates(self):
        """ determine center coordinates of all image sections """
        xstart, xend = self._xwidth*2, self._img.shape[0] - (self._xwidth*2)
        ystart, yend = self._ywidth*2, self._img.shape[1] - (self._ywidth*2)

        x_ = np.arange(xstart, xend, self._xstep)
        y_ = np.arange(ystart, yend, self._ystep)

        return list(np.array(np.meshgrid(x_, y_)).T.reshape(-1, 2))

    def calculate_beta(self, beta_percentile=50, beta_count=200):
        """ calculate beta for all image sections """

        import pyfftw

        if self.method == 'numpy':
            fft = np.fft.fft2
        else:
            pyfftw.forget_wisdom()

        beta_stack = []
        # save image sections
        #for i in np.random.choice(self.coord_N, beta_count):
        for i in range(self.coord_N):
            x, y = self.coords[i]
            self.image_section = self._img[x - self._xwidth:x + self._xwidth + 1, y - self._ywidth:y + self._ywidth + 1].copy()
            #image_section = signal.convolve(image_section, hanning, mode='same')
            self.image_section *= self.hanning

            imbar = np.sum(np.sqrt(self.image_section))

            if self.method == 'numpy':
                fft = np.fft.fft2
            else:
                #pyfftw.forget_wisdom()
                fft = pyfftw.builders.fft2(self.image_section, threads=self._nthread, planner_effort='FFTW_ESTIMATE')
            self.fourier_section = fft(self.image_section)

            fourier_magnitude = np.abs(np.fft.fftshift(self.fourier_section))
            beta_stack.append(fourier_magnitude / imbar)

        result = np.percentile(np.stack(beta_stack), beta_percentile, axis=0)
        #result = np.median(np.stack(beta_stack), axis=0)
        del beta_stack

        if self._debug:
            print('... beta count: {}, beta percentile: {}'.format(beta_count, beta_percentile))
            print('... xwidth: {}, step: {}, NX: {}, NY: {}'.format(self._xwidth, self._xstep, self.NX, self.NY))
            plt.imshow(result)
            plt.show()

        return result

    def _process_section(self, i):
        """ process image denoise process in section i """
        import pyfftw

        x, y = self.coords[i]
        self.image_section[:] = self._img[x - self._xwidth:x + self._xwidth + 1, y - self._ywidth:y + self._ywidth + 1].copy()
        self.image_section *= self.hanning
        imbar = np.sum(np.sqrt(self.image_section))

        if self.method == 'numpy':
            fft = np.fft.fft2
        else:
            fft = pyfftw.builders.fft2(self.image_section, threads=self._nthread, planner_effort='FFTW_ESTIMATE')
        self.fourier_section = fft(self.image_section)

        fourier = np.fft.fftshift(self.fourier_section)
        fourier_magnitude = np.abs(fourier)

        noise = self.beta * imbar
        threshold = noise * self.gamma

        #gate_filter = fourier_magnitude > threshold
        wiener_filter = (fourier_magnitude / threshold) / (1 + (fourier_magnitude / threshold))
        self.fourier_section[:] = fourier * wiener_filter
        self.fourier_section = np.fft.fftshift(self.fourier_section)

        # inverse fourier transform
        if self.method == 'numpy':
            ifft = np.fft.ifft2
        else:
            ifft = pyfftw.builders.ifft2(self.fourier_section, threads=self._nthread, planner_effort='FFTW_ESTIMATE')
        self.image_section = ifft(self.fourier_section)
        final_image = self.hanning * np.abs(self.image_section)

        return final_image

    def clean(self):
        """ calculate denoised image """

        if self._debug:
            print("... Start denoise process by Deforest (2017)")

        start_time = time.time()

        self.beta = self.calculate_beta(self.beta_percentile, self.beta_count)
        new_image = np.zeros_like(self._img)

        import multiprocessing
        with multiprocessing.Pool() as p:
            sections = p.map(self._process_section, range(self.coord_N))

        for i in range(self.coord_N):
            x, y = self.coords[i]
            new_image[x - self._xwidth:x + self._xwidth + 1, y - self._ywidth:y + self._ywidth + 1] += sections[i]

        # adjust outer area
        new_image = new_image[2*self._xwidth:-2*self._xwidth, 2*self._ywidth:-2*self._ywidth]
        new_image = (new_image - new_image.min()) / (new_image.ptp())

        if self._debug:
            print("... %s seconds" % (time.time() - start_time))

        if self._types == 'uint8':
            return (new_image * 255.0).astype(np.uint8)
        else:
            return new_image

# vim:foldmethod=indent:foldlevel=0
