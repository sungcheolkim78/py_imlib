"""
imgLog.py - experimental log for imgFolder

initial: 2019-10-04
"""

import os
import pandas as pd

if ('np' not in dir()): import numpy as np

from imlib.imgfolder import ImgFolder

__author__ = 'Sungcheol Kim <kimsung@us.ibm.com>'
__version__ = '1.0.0'

class ImgLog(ImgFolder):
    """ imgFolder for channel experiment images """

    def __init__(self, dirname, sort=False, debug=False):
        """ initialization """

        super().__init__(dirname, sort=sort, debug=debug)

        self._logfname = dirname + '/log.xlsx'

        if not self.loadLog():
            self._data['wall1'] = self._image._meta['wall1']
            self._data['wall2'] = self._image._meta['wall2']
            self._data['isChannel'] = False
            self._data['range1'] = self._image._meta['range1']
            self._data['range2'] = self._image._meta['range2']
            self._data['hasRange'] = False
            self._data['fangle'] = 0.    # frame angle
            self._data['mangle'] = 0.    # migration angle
            self._data['p'] = 0.         # pressure
            self._data['u'] = 0.         # longitudinal velocity
            self._data['mag'] = '40x'    # magnification
            self._data['filter'] = ''    # filter
            self._data['exp'] = 0.       # exposure
            self._data['location'] = ''  # location
            self._data['D'] = 0.         # diffusion constant
            self._data['Pe'] = 0.        # Peclet number

        self._data = self._data.astype(
                {'D':'float', 'Pe':'float', 'mangle':'float',
                    'hasRange':'bool', 'isChannel':'bool',
                    'exp':'float', 'range1':'int', 'range2':'int',
                    'wall1':'int', 'wall2':'int'})

    def __repr__(self):
        """ show print out message """

        msg = super().__repr__()
        return msg

    def __getitem__(self, fileID):
        self._image = super().__getitem__(fileID)

        p = self._data.at[fileID, 'p']
        if isinstance(p, str) and (p.find(',') > -1):
            p = float(p.replace(',', '.'))
            self._data.at[fileID, 'p'] = p
        if isinstance(p, float) or isinstance(p, np.int64):
            u = 143.9518*p + 44.0784    # TODO in case of condenser chip
            self._data.at[fileID, 'u'] = u
            if self._debug: print('... (p, u) = {}, {}'.format(p, u))
        self._data.at[fileID, 'exp'] = self._image._meta['exposuretime']

        self._image.set_expInfo(magnification=self._data.at[fileID, 'mag'],
                velocity=self._data.at[fileID, 'u'], p=p,
                fangle=self._data.at[fileID, 'fangle'])

        return self._image

    # manage log excel sheet

    def saveLog(self):
        """ save log sheet """

        with pd.ExcelWriter(self._logfname) as writer:
            if self._debug: print('... save to {}'.format(self._logfname))
            self._data.to_excel(writer)

    def loadLog(self):
        """ load log sheet """

        if os.path.isfile(self._logfname):
            if self._debug: print('... load from {}'.format(self._logfname))
            self._data = pd.read_excel(self._logfname, index_col=0)
            return True
        else:
            return False

    # image analysis
    def set_log(self, colname, values, ranges=[]):
        """ set log values for specific colname """

        if len(ranges) == 0:
            ranges = range(len(self._data))
        for i in ranges:
            self._data.at[i, colname] = values
            if self._debug: print('{}: [{}] - {}'.format(i, colname, values))

    def detect_channel(self, fileID=-1, show=True):
        """ find wall information and save in object """

        if fileID > -1:
            self._image = self.getfile(fileID)
        res = self._image.detect_channel(show=show)

        if len(res) > 3:
            self._data.at[self._curidx, 'range1'] = res[2]
            self._data.at[self._curidx, 'range2'] = res[3]
            self._data.at[self._curidx, 'hasRange'] = True
        if len(res) > 1:
            self._data.at[self._curidx, 'wall1'] = res[0]
            self._data.at[self._curidx, 'wall2'] = res[1]
            self._data.at[self._curidx, 'isChannel'] = True
        if len(res) == 1:
            self._data.at[self._curidx, 'isChannel'] = False

        return res

    def analysis_10x(self, fileID, bfileID=-1, wallinfo=[], p=-1, method='gaussian', update=True, padding=0):
        """ find angle and diffusion constant in 10x flu. and bright field images """

        angle = 0.0

        if p > -1:
            self._data.at[self._curidx, 'p'] = p

        if len(wallinfo) == 4:
            self._data.loc[fileID, ['wall1', 'wall2', 'range1', 'range2']] = wallinfo
            print('... fileID: [{}] use wallinfo: {}, ranges: {}'.format(fileID, wallinfo[:2], wallinfo[2:]))
        else:
            if bfileID > -1:
                wallinfo = self.detect_channel(fileID=bfileID)
                wallinfo[0] = wallinfo[0] + padding
                wallinfo[1] = wallinfo[1] - padding
                if len(wallinfo) == 3:
                    self._data.loc[fileID, ['wall1', 'wall2', 'range1']] = wallinfo
                elif len(wallinfo) == 4:
                    self._data.loc[fileID, ['wall1', 'wall2', 'range1', 'range2']] = wallinfo
                else:
                    print('... no wall. Is this [{}] correct image?'.format(bfileID))
                    return
                img = self.__getitem__(bfileID)
                angle = img.detect_angles(show=False)
            print('... fileID: [{}] use wallinfo: {}, ranges: {}, frame angle: {}'.format(fileID, wallinfo[:2], wallinfo[2:], angle))

        # set image information
        self._image = self.__getitem__(fileID)
        self._image.set_wallinfo(self._data.loc[fileID, ['wall1', 'wall2', 'range1', 'range2']])
        self._image.set_expInfo(magnification=self._data.at[fileID, 'mag'],
                velocity=self._data.at[fileID, 'u'], p=self._data.at[fileID, 'p'],
                fangle=self._data.at[fileID, 'fangle'])

        # calculate peak positions
        self._image.fitlines_x(method=method, update=update)
        self._image.showfit_sigmas()
        self._image.showfit_angles()

        # save results
        self._data.at[fileID, 'mangle'] = self._image._meta['mangle']
        self._data.at[fileID, 'D'] = self._image._meta['D']
        self._data.at[fileID, 'Pe'] = self._image._meta['Pe']
        if angle > 0:
            self._data.at[fileID, 'fangle'] = angle

    def analysis_all(self, blist, flist, method='gaussian', update=False):
        """ analaysis migration angle of files in flist with wall info from
        blist """

        if isinstance(blist, int):
            blist = np.zeros_like(np.array(flist)) + blist

        for i in range(len(flist)):
            self.analysis_10x(flist[i], bfileID=blist[i], padding=5,
                    update=update, method=method)

        self.saveLog()

    def showinfo(self, colname='mag', condition='10x'):
        """ show panda data with condition """

        return self._data[self._data[colname] == condition]


# vim:foldmethod=indent:foldlevel=0
