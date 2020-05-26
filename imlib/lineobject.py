"""
lineobject.py - object to keep line profile and related information

date: 20191023 - separated from baselines.py
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy.signal import find_peaks
from tqdm import tnrange

from .rfit import robust_gaussian_fit
from .rfit import robust_gaussian2_fit
from .rfit import robust_gaussian3_fit
from .rfit import robust_gcdf_fit
from .rfit import gaussian, gaussian2, gaussian3, gcdf


__author__ = "Sung-Cheol Kim <kimsung@us.ibm.com>"
__version__ = '1.0.1'


class LineObject(object):
    """ Line Object to keep line profile data """

    def __init__(self, img, x0, y0, x1, y1, label='distance (pixel)', debug=False):
        self._debug = debug

        # original image
        self._img = img

        self.update_coords(x0, y0, x1, y1)

        # fitting 
        self._kfit = []
        self._fit_y1 = []
        self._fit_y2 = []
        self._fit_y3 = []

        self._peaks = []
        self._left_inp = []
        self._right_inp = []

        self._label = label
        self._fittype = None
        self._msg = ""

    @classmethod
    def from_x(cls, img, y0, debug=False):
        return cls(img, 0, y0, img.shape[1], y0, label='x (pixel)', debug=debug)

    @classmethod
    def from_y(cls, img, x0, debug=False):
        return cls(img, x0, 0, x0, img.shape[0], label='y (pixel)', debug=debug)

    def get(self, smooth=0, window='flat', dt=False, integral=False, norm=False):
        """ obtain coordinate and intensity from line profile """

        # smoothing line profile using different convolution window function
        if smooth > 0:
            if self._debug: print('... smooth line with {}'.format(smooth))
            self._z = _smooth(self._z, window_len=smooth, window=window)

        # calculate gradient
        if dt:
            if self._debug: print('... derivative line')
            self._dz = np.gradient(self._z, self._d)
            nz = self._dz.max() if norm else 1.0
            return [self._d, self._dz/nz]

        if integral:
            if self._debug: print('... integral line')
            self._iz = np.cumsum(self._z - np.min(self._z))
            nz = self._iz.max() if norm else 1.0
            return [self._d, self._iz/nz]

        nz = self._z.max() if norm else 1.0
        return [self._d, self._z/nz]

    def get_coords(self):
        return (self._x0, self._y0, self._x1, self._y1)

    def update_coords(self, x0, y0, x1, y1):
        """ set coordiantes and get intensities """
        self._x0, self._y0 = x0, y0
        self._x1, self._y1 = x1, y1

        if self._debug: print('... read line ({}, {}, {}, {})'.format(x0, y0, x1, y1))
        self._x, self._y, self._z, self._d = _getline(self._img, x0, y0, x1, y1)
        self._iz = []
        self._dz = []

        if x0 == x1: self._loc = x0
        elif y0 == y1: self._loc = y0
        else: self._loc = int(np.mean([x0, x1]))

        self._n = len(self._z)
        self._dpi = self._d.ptp()/float(self._n)

    # fitting methods
    def get_peaks(self, dt=False, width=5, prominence=0.005, sort=False, **kwargs):
        """ obtain maximum and minimum peaks """
        smooth = kwargs.pop("smooth", 0)
        _, z = self.get(dt=dt, smooth=smooth)

        p_peaks, _ = find_peaks(z, width=width, prominence=prominence, **kwargs)
        n_peaks, _ = find_peaks(-z, width=width, prominence=prominence, **kwargs)

        # sort from highest peak to loweset
        if sort:
            if self._debug: print('... sort peak indexes')
            idx = np.argsort(z[p_peaks])
            p_peaks = p_peaks[idx[::-1]]
            idx = np.argsort(z[n_peaks])
            n_peaks = n_peaks[idx[::-1]]

        return np.array(p_peaks), np.array(n_peaks)

    def fit_peaks(self, topn=1, **kwargs):
        """ find peak and maximum slope point """
        width = kwargs.pop('width', 5)

        # find positive peaks
        self._p_peaks, _ = self.get_peaks(sort=True, width=width, **kwargs)
        if self._debug: print('... peaks: {}'.format(self._p_peaks))
        topn = np.min([len(self._p_peaks), topn])
        d0 = int(self._d[0])
        self._peaks = self._p_peaks[:topn] + d0

        # find peaks in derivative
        self._p_dpeaks, self._n_dpeaks = self.get_peaks(dt=True, width=width, **kwargs)
        if self._debug:
            print('... dt p peaks: {}'.format(self._p_dpeaks))
            print('... dt n peaks: {}'.format(self._n_dpeaks))

        # connect to slopes
        for i in range(topn):
            v = self._p_peaks[i]

            # check left inplection point
            p_dpeaks = self._p_dpeaks[self._p_dpeaks < v]
            if len(p_dpeaks) == 0:
                p_delta = v + d0
            else:
                p_delta = min(p_dpeaks, key=lambda x: abs(x-v)) + d0
            self._left_inp.append(p_delta)

            # check right inpliction point
            n_dpeaks = self._n_dpeaks[self._n_dpeaks > v]
            if len(n_dpeaks) == 0:
                n_delta = v + d0
            else:
                n_delta = min(n_dpeaks, key=lambda x: abs(x-v)) + d0
            self._right_inp.append(n_delta)

            self._msg = '[{}] peak: {}, + slope: {}, - slope: {}'.format(i, v + d0, p_delta, n_delta)
            if self._debug: print('... ' + self._msg)

        self._fittype = 'peaks'
        self._fit_x1 = self._d
        self._fit_y1 = self._z
        self._left_inp = np.array(self._left_inp)
        self._right_inp = np.array(self._right_inp)

        return self

    def fit_gaussian_from(self, nu=1000, kfit=[]):
        if len(kfit) == 0:
            return self.fit_gaussian(nu=nu, peaks=[], amps=[], sigmas=[], baseline=0.0)
        else:
            return self.fit_gaussian(nu=nu, peaks=[kfit[0]], amps=[kfit[2]], sigmas=[kfit[1]], baseline=kfit[3])

    def _update_fit_result(self, ores, func):
        """ update fitting result """

        if (ores.x[0] < self._d[0]) or (ores.x[0] > self._d[-1]):
            ores.success = False

        if ores.success is not True:
            print('... [{}, {}] fitting failed!'.format(self._x0, self._y0))
            self._peaks = [np.nan]
            self._left_inp = [np.nan]
            self._right_inp = [np.nan]
            return False

        self._kfit = ores.x
        self._peaks = [ores.x[0]]
        self._left_inp = [ores.x[0] - ores.x[1]]
        self._right_inp = [ores.x[0] + ores.x[1]]
        self._fit_x1 = self._d
        self._fit_y1 = func(ores.x, self._d)
        return True

    def fit_gcdf(self, nu=1000.0, mu=-1):
        x, y = self.get(integral=True, norm=True)

        # guess initial values
        if mu == -1:
            mu = np.argmax(self._z)*self._dpi + self._d[0]
        init = [float(mu), 1.0, 0.01, 0.01]

        if self._debug: print('... {}'.format(init))
        ores = robust_gcdf_fit(x, y, nu=nu, initial=init, debug=self._debug)
        if ores.success and self._debug: print('... {}'.format(ores.x))
        if not self._update_fit_result(ores, gcdf): return self

        self._fittype = 'normal cdf'
        self._msg = 'Mean: {:12.4f}    Deviation: {:12.4f}\nBaseline slope: {:12.4f}'.format(ores.x[0], ores.x[1], ores.x[2])

        if self._debug:
            print('... Mean: %f' % ores.x[0])
            print('... Deviation: %f' % ores.x[1])
            print('... Baseline slope: %f' % ores.x[2])

        return self

    def fit_gaussian(self, nu=1000, peaks=[], amps=[], sigmas=[], baseline=0.0):
        """ calculate gaussian fit """
        x, z = self.get(norm=True)

        # guess initial values
        if len(amps) == 0:
            amps = [1.0]
        if len(sigmas) == 0:
            sigmas = [1.0]
        if len(peaks) == 0:
            peaks = [self._d[0] + np.argmax(z)*self._dpi]
        baseline = np.min(z) if baseline == 0.0 else baseline
        init = [peaks[0], sigmas[0], amps[0], baseline]

        if self._debug: print('... Peak, Sigma, Amp, Baseline estimate: {}'.format(init))
        ores = robust_gaussian_fit(x, z, nu=nu, initial=init, debug=self._debug)
        if not self._update_fit_result(ores, gaussian): return self

        self._fittype = 'gaussian'
        self._msg = 'Amplitude: {:12.4f}     Mean: {:12.4f}\nDeviation: {:12.4f}     Baseline: {:12.4f}'.format(ores.x[2], ores.x[0], ores.x[1], ores.x[3])

        if self._debug:
            print('... Amplitude: %f' % ores.x[2])
            print('... Mean: %f' % ores.x[0])
            print('... Deviation: %f' % ores.x[1])
            print('... Baseline: %f' % ores.x[3])

        return self

    def fit_gaussian2(self, nu=30.0, peaks=[], amps=[], sigmas=[], baseline=0.0, mode='single'):
        """ calculate double gaussian fit """
        if len(peaks) == 2:
            if len(amps) == 0: amps = [1.0, 1.0]
            if len(sigmas) == 0: sigmas = [1.0, 1.0]

        kfit = [peaks[0], peaks[1], sigmas[0], sigmas[1], amps[0], amps[1], baseline]
        ores = robust_gaussian2_fit(self._d, self._z, nu=nu, initial=kfit, debug=self._debug)
        self._kfit = ores
        self._hasfit = True
        self._fittype = 'gaussian3'

        if mode == 'single':
            self._fit_y1 = gaussian([ores[4], ores[0], ores[2], ores[6]], self._d)
            self._fit_y2 = gaussian([ores[5], ores[1], ores[3], ores[6]], self._d)

            self._fit_x1 = self._d[:int(ores[0] + 3.5 * ores[2])]
            self._fit_y1 = self._fit_y1[:int(ores[0] + 3.5 * ores[2])]

            self._fit_x2 = self._d[int(ores[1] - 3.5 * ores[3]):]
            self._fit_y2 = self._fit_y2[int(ores[1] - 3.5 * ores[3]):]
        else:
            self._fit_y1 = gaussian2(ores, self._d)
            self._fit_x1 = self._d

        if self._debug:
            print('... [1] A: %.4f mu: %.4f sigma: %.4f base: %.1f' % (ores[4], ores[0], ores[2], ores[6]))
            print('... [2] A: %.4f mu: %.4f sigma: %.4f base: %.1f' % (ores[5], ores[1], ores[3], ores[6]))

        return self._kfit

    def fit_gaussian3(self, nu=30.0, peaks=[], amps=[], sigmas=[], baseline=0.0, mode='single'):
        """ calculate triple gaussian fit """
        if len(peaks) == 3:
            if len(amps) == 0: amps = [1.0, 1.0, 1.0]
            if len(sigmas) == 0: sigmas = [1.0, 1.0, 1.0]

        kfit = [amps[0], peaks[0], sigmas[0], amps[1], peaks[1], sigmas[1], amps[2], peaks[2], sigmas[2], baseline]
        ores = robust_gaussian3_fit(self._d, self._z, nu=nu, initial=kfit, debug=self._debug)
        self._kfit = ores
        self._hasfit = True
        self._fittype = 'gaussian3'

        if mode == 'single':
            self._fit_y1 = gaussian([ores[0], ores[1], ores[2], ores[9]], self._d)
            self._fit_y2 = gaussian([ores[3], ores[4], ores[5], ores[9]], self._d)
            self._fit_y3 = gaussian([ores[6], ores[7], ores[8], ores[9]], self._d)

            self._fit_x1 = self._d[:int(ores[1] + 3.5 * ores[2])]
            self._fit_y1 = self._fit_y1[:int(ores[1] + 3.5 * ores[2])]

            self._fit_x2 = self._d[int(ores[4] - 3.5 * ores[5]):int(ores[4] + 3.5 * ores[5])]
            self._fit_y2 = self._fit_y2[int(ores[4] - 3.5 * ores[5]):int(ores[4] + 3.5 * ores[5])]

            self._fit_x3 = self._d[int(ores[7] - 3.5 * ores[8]):]
            self._fit_y3 = self._fit_y3[int(ores[7] - 3.5 * ores[8]):]
        else:
            self._fit_y1 = gaussian3(ores, self._d)
            self._fit_x1 = self._d

        if self._debug:
            print('[1] A: %.4f mu: %.4f sigma: %.4f base: %.4f' % (ores[0], ores[1], ores[2], ores[9]))
            print('[2] A: %.4f mu: %.4f sigma: %.4f base: %.4f' % (ores[3], ores[4], ores[5], ores[9]))
            print('[3] A: %.4f mu: %.4f sigma: %.4f base: %.4f' % (ores[6], ores[7], ores[8], ores[9]))

        return self._kfit

    def fit_sigmoid(self):
        """ fit line with sigmoid function """
        pass

    def getfits_x(self, ylist, x0=-1, x1=-1, method='gaussian'):
        """ fit multiple lines """

        if x0 == -1: x0 = 0
        if x1 == -1: x1 = self._img.shape[1]
        if method == 'gcdf':
            func = self.fit_gcdf
        else:
            func = self.fit_gaussian
        res = np.zeros((len(ylist),6))

        for i in tnrange(len(ylist)):
            self.update_coords(x0, ylist[i], x1, ylist[i])

            if method == 'gcdf':
                self.fit_gcdf()
            else:
                self.fit_gaussian()

            res[i,:] = [i, self._peaks[0], self._peaks[0] - self._left_inp[0], 
                    self._peaks[0] - self._right_inp[0], 
                    self._left_inp[0], self._right_inp[0]]

        return res

    def plot(self, norm=False, z=True, dt=False, integral=False, msg="", ax=None, **kwargs):
        if ax is None: ax = plt.gcf().gca()
        if self._fittype is not None: norm=True

        label = kwargs.pop('label', 'raw')
        if z:         # intensity
            x, y = self.get(norm=norm)
            ax.plot(x, y, label=label, **kwargs)
        if dt:        # derivative
            x, y = self.get(dt=True, norm=norm)
            ax.plot(x, y, label='dz', linestyle='dashed')
        if integral:  # integral
            x, y = self.get(integral=True, norm=norm)
            ax.plot(x, y, label='integral', linestyle='dashed')
        ax.set_xlim(self._d.min(), self._d.max())
        ax.set_xlabel(self._label)
        ylab = 'Nor. Intensity' if norm else 'Intensity'
        ax.set_ylabel(ylab)

        if (self._msg != "") and (msg == ""):
            msg = self._msg
        ax.text(0.1, -0.3, msg, ha='left', transform=ax.transAxes)

        if self._fittype is not None:
            ax.plot(self._fit_x1, self._fit_y1, '--', label='fit')

        if self._fittype == 'peaks':
            ax.plot(self._d[self._p_peaks], self._z[self._p_peaks], "o", label='peak')
            ax.plot(self._d[self._p_dpeaks], self._z[self._p_dpeaks], "x")
            ax.plot(self._d[self._n_dpeaks], self._z[self._n_dpeaks], "+")
            d0 = int(self._d[0])
            if len(self._peaks) > 0:
                ax.vlines(self._d[self._peaks-d0], np.min(self._z), self._z[self._peaks-d0], alpha=0.5)
                ax.vlines(self._d[self._left_inp-d0], np.min(self._z), self._z[self._left_inp-d0], alpha=0.5)
                ax.vlines(self._d[self._right_inp-d0], np.min(self._z), self._z[self._right_inp-d0], alpha=0.5)

        ax.legend(loc='best')


def _getline(img, x0, y0, x1, y1, num_min=50):
    """ get line profile data from image 

    Return
    ------
    [x, y, zi, di]
        x: array of x coordinates
        y: array of y coordinates
        zi: array of intensities
        di: array of distances
    """

    zi = []
    [x0, x1] = np.clip([x0, x1], 0, img.shape[1] - 1)
    [y0, y1] = np.clip([y0, y1], 0, img.shape[0] - 1)

    if x0 == x1:
        dis = np.abs(y1 - y0 + 1)
        if dis > num_min:
            zi = np.asarray(img[int(y0):int(y1)+1, int(x0)])
            return [np.zeros(dis)+x0, np.arange(y0,y1+1), zi, np.arange(y0,y1+1)]

    elif y0 == y1:
        dis = np.abs(x1 - x0 + 1)
        if dis > num_min:
            zi = np.asarray(img[int(y0), int(x0):int(x1)+1])
            return [np.arange(x0,x1+1)+x0, np.zeros(dis)+y0, zi, np.arange(x0,x1+1)]

    dis = np.sqrt((x0 - x1)*(x0 - x1) + (y0 - y1)*(y0 - y1))
    num = max(int(dis) - 1, num_min)

    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    zi = scipy.ndimage.map_coordinates(img, np.vstack((y, x)))
    #di = np.sqrt((x - x[0])**2 + (y - y[0])**2)
    di = np.linspace(0, dis, num)

    return [x, y, zi, di]


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
    return y[(window_len//2):-(window_len//2)]


# vim:foldmethod=indent:foldlevel=0
