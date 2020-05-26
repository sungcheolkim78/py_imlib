"""
base_lines.py - microscopy images class

update: 20191001 - modify gaussian fit function
"""

import os
import sys
from tqdm import tqdm_notebook

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from scipy.signal import argrelmax

# local library
from .rfit import robust_line_fit
from .rfit import robust_inverseabs_fit
from .rfit import gaussian, gaussian2, gaussian3, line, inverseabs

from .base_filters import ImageFilter
from .lineobject import LineObject
from .lineobject import _smooth

__author__ = 'Sungcheol Kim <kimsung@us.ibm.com>'
__version__ = '1.1.0'


class ImageLine(ImageFilter):
    """ Image based on channel experiments  - lineprofile and migration angle detection """

    def __init__(self, objects, method='tifffile', debug=False, **kwargs):
        """ initialization """

        super().__init__(objects, method=method, debug=debug, **kwargs)
        """ TifFile class initialization """

        # for graphic representation
        self._ax1 = None
        self._ax2 = None

        # fitting info
        self._peakdatafname = '.{}_peakinfo.csv'.format(self._meta['basename'])
        if os.path.isfile(self._peakdatafname):
            if self._debug: 
                print('... read peak information: {}'.format(self._peakdatafname))
            self._peakdata = pd.read_csv(self._peakdatafname, index_col=0)
        else:
            self._peakdata = pd.DataFrame()

        self._lineobject = []

        self._kfit = []
        self._baseline = 0.0
        self._beaminfo = []
        self._beamdirection = 'right'

        # full width injection case
        self._smoothing = 13
        self._searchrange = 10

    def __repr__(self):
        """ representation """

        msg = super().__repr__()

        msg += '-'*50 + '\n'
        msg += '... Wall positions: ({}, {})\n'.format(self._meta['wall1'], self._meta['wall2'])
        msg += '... Array positions: ({}, {})\n'.format(self._meta['range1'], self._meta['range2'])
        msg += '... Migration Angle: {:.4f} [deg]\n'.format(self._meta['mangle'])
        msg += '... Frame Angle: {:.4f} [deg]\n'.format(self._meta['fangle'])
        msg += '... Diffusion constant: {:.4f} [um2/s]\n'.format(self._meta['D'])
        msg += '... Peclet number: {:.4f}\n'.format(self._meta['Pe'])
        msg += '... Pressure: {:.4f} [bar]\n'.format(self._meta['p'])
        msg += '... Velocity: {:.5f} [um/s]\n'.format(self._meta['u'])
        msg += '... Particle size: {:.4f} [nm]\n'.format(self._meta['psize'])

        return msg

    def set_wallinfo(self, wallinfo=[0, 512, 0, 512], show=False):
        """ manually set wall information """

        if len(wallinfo) == 4:
            self._meta.update_wall(wallinfo)
        else:
            print('... wallinfo = [wall1, wall2, range1, range2]')
            return

    def set_expInfo(self, magnification=None, velocity=-1, p=-1, fangle=0.0, psize=-1, ccd_length=16.0):
        """ set experimental values """
        if isinstance(magnification, str):
            self._meta.update_mag(magnification)
            self._meta.update_wall([0, 0, 0, 0])

        if velocity > -1: self._meta['u'] = velocity
        if p > -1: self._meta['p'] = p
        if fangle != 0: self._meta['fangle'] = fangle
        if psize > -1: self._meta['psize'] = psize
        # TODO add bulk diffusion contant

    # line profile
    def getline_obj(self, frame=-1, coords=None, dtypes='orig'):
        """ generate line object using coordinates """

        if frame == -1: img = self.tmean(dtypes=dtypes)
        else: img = self.getframe(frame=frame, dtypes=dtypes)
        self._lineobject = _getline_obj(img, coords=coords)
        return self._lineobject

    def getline_x(self, frame=-1, y=-1, dtypes='orig', **kwargs):
        """ get line profile along x axis """

        if frame == -1: img = self.tmean(dtypes=dtypes)
        else: img = self.getframe(frame=frame, dtypes=dtypes)
        return self._getline_x(img, y=y, **kwargs)

    def getline_y(self, frame=-1, x=-1, dtypes='orig'):
        """ get line profile along y axis """

        if frame == -1: img = self.tmean(dtypes=dtypes)
        else: img = self.getframe(frame=frame, dtypes=dtypes)
        return self._getline_y(img, x=x)

    def _getline_x(self, img, y=-1, ignore_wall=False):
        """ get line profile along x axis using coordinates """

        if (not ignore_wall):
            xs, xf = self._meta['wall1'], self._meta['wall2']
        return _getline_obj(img, coords=[xs, y, xf, y])

    def _getline_y(self, img, x=-1, ignore_wall=False):
        """ get line profile along y axis using coordinates """

        if (not ignore_wall):
            ys, yf = self._meta['range1'], self._meta['range2']
        return _getline_obj(img, coords=[x, ys, x, yf])

    def getlines_x_shadow(self, locations, raw=False):
        """ get lines with background subtraction """
        results = []
        results_raw = []
        for i in range(len(locations)):
            line = _zprofile(self.tmean(), locations[i])
            results_raw.append(np.copy(line))
            line -= self._findShadowline(locations[i])
            results.append(_smooth(line, window_len=self._smoothing))
        if raw:
            return (results, results_raw)
        else:
            return results

    # fit line profile with various methods
    def fitline_x(self, frame=-1, y=-1, method='peaks', **kwargs):
        """ fitting line intensity profile along x axis """

        line_obj = self.getline_x(frame=frame, dtypes='float', y=y)
        self._lineobject = line_obj
        return self._fitline(line_obj, method=method, **kwargs)

    def fitline_y(self, frame=-1, x=-1, method='peaks', **kwargs):
        """ fitting line intensity profile along y axis """

        line_obj = self.getline_y(frame=frame, dtypes='float', x=x)
        self._lineobject = line_obj
        return self._fitline(line_obj, method=method, **kwargs)

    def _fitline_x(self, img, y=-1, method='peaks', **kwargs):
        line_obj = self._getline_x(img, y=y)
        return self._fitline(line_obj, method=method, **kwargs)

    def _fitline_y(self, img, x=-1, method='peaks', **kwargs):
        line_obj = self._getline_y(img, x=x)
        return self._fitline(line_obj, method=method, **kwargs)

    def _fitline(self, line_obj, method='gaussian', **kwargs):
        """ get line profile and find peak """

        nu = kwargs.pop('nu', 100)
        if method == 'gaussian':
            if self._debug: print('... gaussian fit')
            line_obj.fit_gaussian_from(nu=nu, kfit=self._kfit, **kwargs)
        elif method == 'gaussian2':
            if self._debug: print('... double gaussian fit')
            line_obj.fit_gaussian2_from(nu=nu, kfit=self._kfit, **kwargs)
        elif method == 'gaussian3':
            if self._debug: print('... triple gaussian fit')
            line_obj.fit_gaussian3_from(nu=nu, kfit=self._kfit, **kwargs)
        elif method == 'peaks':
            if self._debug: print('... peak find fit')
            line_obj.fit_peaks(**kwargs)
        elif method == 'gcdf':
            if self._debug: print('... gaussian cdf fit')
            line_obj.fit_gcdf(nu=nu)

        return line_obj

    def fitlines_x(self, locs=-1, method='gaussian', update=False, **kwargs):
        return self._fitlines_x(self.tmean(), locs=locs, method=method, update=update, **kwargs)

    def _fitlines_x(self, img, locs=-1, method='gaussian', update=False, **kwargs):
        """ get peak position datasheet at locs """

        # set all y
        if locs == -1:
            locs = range(int(self._meta['range1']), int(self._meta['range2']))

        # read from cache
        if (not update) and os.path.isfile(self._peakdatafname):
            if self._debug: print('... read from %s' % self._peakdatafname)
            self._peakdata = pd.read_csv(self._peakdatafname, index_col=0)
        else:
            if self._debug: print('... create %s' % self._peakdatafname)
            d = np.zeros((img.shape[1], 6)) - 1
            self._peakdata = pd.DataFrame(d, index=range(img.shape[1]), columns=['loc', 'peak', 'delta', 'rdelta', 'l_inp', 'r_inp'])

        # iterate over all range
        count = 0
        keep_debug = self._debug
        self._debug = False
        for i in tqdm_notebook(locs):
            if update or (self._peakdata['loc'].loc[i] == -1):
                line_obj = self._fitline(self._getline_x(img, y=i), method=method, **kwargs)
                self._peakdata['loc'].loc[i] = i
                try:
                    self._peakdata['peak'].loc[i] = line_obj._peaks[0]
                    self._peakdata['delta'].loc[i] = line_obj._peaks[0] - line_obj._left_inp[0]
                    self._peakdata['rdelta'].loc[i] = line_obj._right_inp[0] - line_obj._peaks[0]
                    self._peakdata['l_inp'].loc[i] = line_obj._left_inp[0]
                    self._peakdata['r_inp'].loc[i] = line_obj._right_inp[0]
                except: pass
                count += 1

        # save if updated
        if count > 0:
            if self._debug: print('... save to %s' % self._peakdatafname)
            self._peakdata.to_csv(self._peakdatafname)

        self._debug = keep_debug
        return self._peakdata

    # show line profiles
    def _show_lineobj(self, line_obj, msg='', save=False, **kwargs):
        """ show image and line profile """

        x0, y0, x1, y1 = line_obj.get_coords()
        _, z = line_obj.get()

        plt.clf()
        fig = plt.figure(figsize=(11, 5))

        # plot image with colormap bar
        self._ax1 = fig.add_subplot(121)
        self._meta._zrange = [z.min(), z.max()]

        self._showimage(line_obj._img, autorange=False, wall=True, ax=self._ax1)
        self._ax1.plot([x0, x1], [y0, y1], 'ro-')
        self._ax1.annotate(str(line_obj._loc), xy=(x1+3, y1), va='top', ha='left', color='red')

        # plot line profile
        self._ax2 = plt.axes([0.62, 0.32, 0.35, 0.55])
        line_obj.plot(msg=msg, ax=self._ax2, **kwargs)

        if save:
            savename = self._meta['basename'] + '_rfit_l{}_{}.pdf'.format(line_obj._loc, method)
            plt.savefig(savename, dpi=200)
            if self._debug: print('... save to {}'.format(savename))

    def showline_x(self, frame=-1, y=-1, dtypes='orig', msg='', **kwargs):
        self._lineobject = self.getline_x(frame=frame, y=y,  dtypes=dtypes)
        self._show_lineobj(self._lineobject, msg=msg, **kwargs)

    def showfit_x(self, frame=-1, y=-1, method='gaussian', msg='', **kwargs):
        self._lineobject = self.fitline_x(frame=frame, y=y, method=method)
        self._show_lineobj(self._lineobject, msg=msg, **kwargs)

    def showfit_peaks(self, ranges=[], method='gaussian', ax=None):
        """ show peakdata with image """

        if len(self._peakdata) == 0:
            self.fitlines_x(method='gaussian')

        if len(ranges) == 2:
            self._meta['range1'] = ranges[0]
            self._meta['range2'] = ranges[1]

        if ax is None:
            plt.clf()
            fig = plt.figure(figsize=(10, 5))
            ax = plt.gcf().gca()

        # plot image
        self._showimage(self.tmean(), frameNumber=False, ax=ax, wall=True)

        # plot transition
        tmp = self._peakdata.iloc[self._meta['range1']: self._meta['range2']]
        x = tmp['loc']
        y = tmp['peak']
        dy1 = tmp['l_inp']
        dy2 = tmp['r_inp']
        ax.plot(dy1, x, '.', color='gray', markersize=1, alpha=0.8, label='')
        ax.plot(dy2, x, '.', color='gray', markersize=1, alpha=0.8, label=r'$\sigma$')
        ax.plot(y, x, '.', color='red', markersize=1, label='peak', alpha=0.8)
        ax.legend(loc='best')

    def showfit_angles(self, ranges=[], method='gaussian', show=True, save=True, **kwargs):
        """ show peak positions and calculate angle """

        # calculate migration angle
        if len(ranges) == 2:
            self._meta['range1'] = ranges[0]
            self._meta['range2'] = ranges[1] 

        if len(self._peakdata) == 0:
            self.fitlines_x(method=method)

        tmp = self._peakdata.iloc[self._meta['range1']: self._meta['range2']]
        x = np.asarray(tmp['loc'])
        y = np.asarray(tmp['peak'].ffill())
        if self._debug: print('... initial: {}, {}'.format(y.ptp()/x.ptp(), y[0]))
        kr = robust_line_fit(x, y, nu=100.0, debug=self._debug, initial=[y.ptp()/x.ptp(), y[0]], **kwargs)
        if self._debug: print('... final: {}, {}'.format(kr.x[0], kr.x[1]))
        yr = line(kr.x, x)
        self._meta['mangle'] = np.arctan(kr.x[0]) * 180.0 / np.pi
        self._meta.save()

        if show:
            # plot peak positions
            plt.clf()
            fig = plt.figure(figsize=(11, 5))

            ax1 = fig.add_subplot(121)
            self.showfit_peaks(ranges=ranges, ax=ax1)

            ax2 = plt.axes([0.62, 0.32, 0.35, 0.55])

            # plot peak position lines
            ax2.plot(x, y)
            ax2.plot(x, yr, '--', label='fit')
            ax2.set_xlabel('locations [pixel]')
            ax2.set_ylabel('peak positions [pixel]')

            msg = 'Shift: {:8.4f} over {:3d}     Slope: {:12.4f}\nAngle: {:12.4f} [deg]      y0: {:12.4f}'.format(yr.ptp(), self._meta['range2'] - self._meta['range1'], kr.x[0], self._meta['mangle'], kr.x[1])
            ax2.text(0.1, -0.4, msg, ha='left', transform=ax2.transAxes)
            ax2.legend(loc='best')

        if show and save:
            savename = self._meta['basename'] + '_angle.pdf'
            if self._debug: print('... save to %s' % savename)
            plt.savefig(savename, dpi=300)
            plt.show()

    def showfit_sigmas(self, ranges=[], show=True, save=True, colname="delta", **kwargs):
        """ show sigma and calculate diffusion coefficient """

        if len(ranges) == 2:
            self._meta['range1'] = ranges[0]
            self._meta['range2'] = ranges[1]

        if len(self._peakdata) == 0:
            self.fitlines_x()

        tmp = self._peakdata.iloc[self._meta['range1']: self._meta['range2']]
        x = np.asarray(tmp['loc'])
        y = np.asarray(tmp[colname].ffill())
        if self._debug: print('... initial: {}, {}'.format(y.ptp()/x.ptp(), y[0]))
        kr = robust_line_fit(x, y**2, nu=300.0, debug=self._debug, initial=[y.ptp()/x.ptp(), y[0]], **kwargs)
        if self._debug: print('... final: {}, {}'.format(kr.x[0], kr.x[1]))
        yr = line(kr.x, x)

        self._meta['Pe'] = 2.0 * self._meta['channelWidth'] / kr.x[0]
        self._meta['D'] = self._meta['u'] * self._meta['arrayWidth'] / self._meta['Pe']     # um^2/s
        self._meta.save()

        if show:
            plt.clf()
            fig = plt.figure(figsize=(11, 5))

            # plot peak positions
            ax1 = fig.add_subplot(121)
            self.showfit_peaks(ranges=ranges, ax=ax1)

            # plot peak position lines
            ax2 = plt.axes([0.62, 0.32, 0.35, 0.55])

            ax2.plot(x, y**2, label=r'$\sigma^2$')
            ax2.plot(x, yr, '--', label='fit')
            ax2.set_xlabel(r'locations [pixel]')
            ax2.set_ylabel(r'sigma^2 [pixel^2]')

            msg = 'D: {:8.4f} [um2/s]     Peclet Number: {:8.4f}\nSlope: {:13.4f}        y0: {:12.4f}'.format(self._meta['D'], self._meta['Pe'], kr.x[0], kr.x[1])
            ax2.text(0.1, -0.4, msg, ha='left', transform=ax2.transAxes)
            ax2.legend(loc='best')

        if show and save:
            savename = self._meta['basename'] + '_sigma.pdf'
            if self._debug: print('... save to %s' % savename)
            plt.savefig(savename, dpi=300)
            plt.show()

    # show multiple line profiles
    def showlines(self, lines=-1, dir='y', log=False, window=[0, 0], save=False, fit=False, smooth=False):
        """ plot line profiles at multiple locations """

        if lines == -1:
            if dir == 'y':
                lines = range(0, self._width, 50)
            else:
                lines = range(0, self._height, 50)

        tmp = self.tmean()

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
                    ax2.plot(range(wl, wr + 1), _smooth(tmp[wl:wr, l]), label=l, color=color)
                    res_line[i, :] = _smooth(tmp[wl:wr, l])[:-1]
                else:
                    ax2.plot(range(wl, wr), tmp[wl:wr, l], label=l, color=color)
                    res_line[i, :] = tmp[wl:wr, l]
                ax1.annotate(l, xy=(l, wl), va='bottom', color='white')
                print("... Loc: %i Peak: %i Value: %f" % (l, np.argmax(tmp[wl:wr, l]), np.max(tmp[wl:wr, l])))
                if fit:
                    out = _fitGaussian(np.range(wl, wr), tmp[wl:wr, l], baseline=self._baseline)
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
                    ax2.plot(range(wl, wr + 1), _smooth(tmp[l, wl:wr]), label=l, color=color)
                    res_line[i, :] = _smooth(tmp[l, wl:wr])[:-1]
                else:
                    ax2.plot(range(wl, wr), tmp[l, wl:wr], label=l, color=color)
                    res_line[i, :] = tmp[l, wl:wr]
                ax1.annotate(l, xy=(wl, l), ha='right', color='white')
                print("... Loc: %i Peak: %i Value: %f" % (l, np.argmax(tmp[l, wl:wr]), np.max(tmp[l, wl:wr])))
                if fit:
                    out = _fitGaussian(np.arange(wl, wr), tmp[l, wl:wr], baseline=self._baseline)
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

    def showlines_allx(self, bg=0.0, bgstd=0.0, window=None):
        """ show line profile along x axis """
        _plotAxis(self.tmean(), 0, background=bg, backstd=bgstd, window=window)

    def showlines_ally(self, bg=0.0, bgstd=0.0, window=None):
        """ show line profile along y axis """
        _plotAxis(self.tmean(), 1, background=bg, backstd=bgstd, window=window)

    # experimental data
    def detect_channel(self, window=11, compareRatio=0.2, minimumIntensity=0.1, angle=0.0, show=True):
        """ find channel wall positions """

        if angle > 0:
            img = _rotate_bound(self.tmean('float'), angle)
        else:
            img = self.tmean(dtypes='float')
        result = _find_channel(img, window=window, compareRatio=compareRatio, minimumIntensity=minimumIntensity)
        self._meta.update_wall(result)

        if show:
            ax = plt.gcf().gca()
            self._showimage(img, simple=False, ax=ax, wall=True)

        return result

    def detect_angles(self, frame=-1, method1='canny', method2=None, min_length=100, show=True):
        """ find angle of camera based on line properties """

        # preprocess image
        self.reverse(frame=frame)
        img = self.filters(frame=frame, method=method1)
        if method2 is not None:
            img = self.threshold(frame=frame, method=method2, show=False)

        # detect lines
        res = _detect_angles(img, min_length=min_length, show=show)

        # reverse filtered image
        self.reverse(frame=frame)

        return res


    # adjust line profiles
    def _find_shadowline_x(self, img, y=-1, xf=-1, show=True):
        """ find shadow line using inverse abs function """

        wallinfo = self._wallinfo

        if len(wallinfo) < 2:
            raise ValueError('... need wall information: {}'.format(wallinfo))

        x, z = self._getline_x(img, y=y, ignore_wall=True).get()

        xf = len(x) if xf == -1 else xf

        # adjust background
        xmin1 = np.argmin(_smooth(z[:wallinfo[0]], window_len=7))
        xmin2 = np.argmin(_smooth(z[wallinfo[1]:xf], window_len=7)) + wallinfo[1]
        zmin1, zmin2 = z[xmin1], z[xmin2]
        background = (zmin2 - zmin1)/(xmin2 - xmin1) * (x - xmin1) + zmin1

        # peak_position = np.argmax(xline)
        # if peak_position < self._peak[0] or peak_position > self._peak[1]:
        #    return np.zeros_like(xline)

        # fit with right side of wall
        zRight = z[wallinfo[1]:xf] - background[wallinfo[1]:xf]
        xRight = x[wallinfo[1]:xf]
        kr = robust_inverseabs_fit(xRight, zRight, initial=[zRight.max(), xRight[0], 1.0], verb=self._debug)
        result = inverseabs(kr.x, x) + background

        if show:
            plt.plot(x, z, label='raw')
            plt.plot(x, result, label='shadow')
            plt.plot(x, background, label='background')
            msg = 'y: {} back: {:.4f}, {:.4f} \nk: {:.2f} {:.2f} {:.2f}'.format(y, zmin1, zmin2, kr.x[0], kr.x[1], kr.x[2])
            plt.annotate(msg, xy=(0, result.max()))
            plt.vlines(wallinfo[:2], z.min(), z.max(), linestyles='dashed', alpha=0.5)
            plt.legend(loc='best')

        return (result, background)

    def _inflectionpoints(self, xdata, verb=False):
        """ find inflection points in lineprofile """

        x_smooth = _smooth(xdata, window_len=self._smoothing)
        dx = np.gradient(x_smooth)
        dx_smooth = _smooth(dx, window_len=self._smoothing)
        wall1 = self._wall1 + self._wallpos
        wall2 = self._wall2 + self._wallpos
        infmax = np.argmax(dx_smooth[wall1:wall2]) + wall1 - 1
        (localmaxs, ) = argrelmax(dx_smooth[wall1:wall2], order=self._searchrange)
        if verb:
            print('... smoothing: %i, order: %i' % (self._smoothing, self._searchrange))
            print('... find %i local maximums' % len(localmaxs))

        # check wall
        if np.abs(infmax - wall2) < 0.1*self._arrayWidth:
            # 10% of array width
            infmax = _find_before(localmaxs + wall1, infmax)

        return (localmaxs+wall1, infmax)

    def plot_lines_3d(self, frame=-1, locations=[]):
        """ plot line profile in 3d """

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.collections import PolyCollection
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        if len(locations) == 0:
            locations = range(0, self._height, 10)

        lines = []
        for i in locations:
            lines.append(self.getline_x(frame=-1, y=i).get()[1])

        if len(self._wallinfo) > 1:
            xs = range(self._wallinfo[0], self._wallinfo[1])
        else:
            xs = range(0, self._width)

        verts = []
        for i in range(len(locations)):
            verts.append(list(zip(xs, lines[i])))

        poly = PolyCollection(verts)
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=locations, zdir='y')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_zlabel('Intensity')
        ax.set_xlim3d(np.min(xs), np.max(xs))
        ax.set_ylim3d(np.max(locations), np.min(locations))
        ax.set_zlim3d(0, np.max(lines))
        plt.show()


def _getline_obj(img, coords=None, debug=False):
    """ get line profile from (x0, y0) to (x1, y1) """

    if coords is not None:
        x0, y0, x1, y1 = coords
    else:
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]

    if coords[1] == -1:
        y1 = y0 = img.shape[0]//2
    if coords[0] == -1:
        x1 = x0 = img.shape[1]//2

    return LineObject(img, x0, y0, x1, y1, debug=debug)


def _detect_angles(threshold_image, min_length=100, max_line_gap=10, threshold=50, show=True, debug=False):
    """ check rotation of image - detect edges and lines and calculate angle """

    if 'cv2' not in dir(): import cv2

    lines = cv2.HoughLinesP(threshold_image, 0.1, np.pi / 720.0, threshold, None, min_length, max_line_gap)
    if debug: print(lines)
    res = []

    if show:
        fig = plt.figure(figsize=(11, 5))
        plt.imshow(threshold_image)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = math.atan2(x1-x2, y1-y2) * 180.0/math.pi
        x0 = (x1 + x2)*0.5
        y0 = (y1 + y2)*0.5
        if show:
            plt.plot((x1, x2), (y1, y2))
            plt.annotate('{:.2f}'.format(theta), xy=(x0, y0), va='top', color='white')
        res.append(theta)

    if show:
        plt.xlim(0, threshold_image.shape[1])
        plt.ylim(threshold_image.shape[0], 0)
        plt.show()

    return np.median(np.array(res))


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
    localmaxs = argrelmax(_smooth(line), order=10)
    for lm in localmaxs[0]:
        print("... local maximums: %i " % lm)

    # pattern recognition 2
    dline = _smooth(line[1:] - line[:-1])
    localmaxds = argrelmax(np.abs(dline), order=15)
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


def _find_after(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    (idx, ) = np.where(a - a0 > 0)
    return a[idx.min()]


def _find_before(a, a0):
    (idx, ) = np.where(a - a0 < 0)
    return a[idx.max()]


def _zprofile(imgMean, location, width=2):
    if not isinstance(location, int):
        raise ValueError("location should be integer number")
    ymax = min(location + width, imgMean.shape[0])
    ymin = max(0, location-width)
    temp = imgMean[ymin:ymax, :]
    return np.sum(temp, axis=0)/(2.0*width+1.0)


def _find_channel(image, window=11, compareRatio=0.2, minimumIntensity=0.1, debug=True):
    """
    find_channel

    Parameters:
        image - image source
        window - smoothing window
        compareRatio - criteria for intensity variation of two walls
        minimumIntensity - boundary delta value of intensity

    Return:
    """

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
    # print(image.shape)
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
            x0, x1 = dx_max+edge_pixel+dx//2, dx_min+edge_pixel+dx//2
        else:
            x0, x1 = dx_min+edge_pixel+dx//2, dx_max+edge_pixel+dx//2
        width = x1 - x0
        if debug: print("... find wall x0, x1 and width: %i, %i, %i" % (x0, x1, width))
    else:
        if debug:
            print("... fail to find channel wall")
            print("... dx_max: %i dI_max: %.3f" % (dx_max, dI_max))
            print("... dx_min: %i dI_min: %.3f" % (dx_min, dI_min))
        return [0, width, 0, height]

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
                y0, y1 = dy_min+dy//2, dy_max+dy//2
            else:
                y0, y1 = dy_max+dy//2, dy_min+dy//2
            if debug: print("... three channel area: %i, %i" % (y0, y1))
            return [x0, x1, y0, y1]
        else:
            y0 = dy_max
            if debug:
                print("... two channel area: %i" % y0)
                print("... dy_min: %i, dI_min: %.3f" % (dy_min, dI_min))
            return [x0, x1, 0, y0]
    elif np.abs(dI_min) > minimumIntensity:
        y0 = dy_min
        if debug:
            print("... two channel area: %i" % y0)
            print("... dy_max: %i, dI_max: %.3f" % (dy_max, dI_max))
        return [x0, x1, y0, height]
    else:
        if debug:
            print("... only channel")
            print("... dy_max: %i, dI_max: %.3f" % (dy_max, dI_max))
            print("... dy_min: %i, dI_min: %.3f" % (dy_min, dI_min))
        return [x0, x1, 0, height]


def _rotate_bound(image, angle):
    """ rotate image with paddings """

    if ('cv2' not in sys.modules) or ('cv2' not in dir()):
        import cv2

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# vim:foldmethod=indent:foldlevel=0
