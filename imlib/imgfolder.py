#!/usr/local/bin/python
"""
imgfolder.py - tif folder iterator class

date - 2017-04-02
date - 2017-04-04 - add cmap change
date - 2017-04-06 - add line fitting
date - 2017-05-11 - organize functions
date - 2017-08-31 - choose PIVTif, ChannelTif, TifFile
date - 2018-02-22 - add detecttrace
date - 2018-03-16 - clean up and add preview
"""
import glob
from tqdm import tnrange
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imlib.base import ImageBase
#from imlib.base_filters import ImageFilter
#from imlib.base_lines import ImageLine
from imlib.base_features import ImageFeature

from .base import _resize_image

__author__ = 'Sung-Cheol Kim'
__version__ = '1.2.1'

ImageObject = ImageFeature


class ImgFolder(object):
    """
    Folder object that shows tif file and allow shift to the next file and folders

    Note:

    Args:
        dirname: directory name
        file_format: ['tif', 'avi'] file extension
        file_type: ['Channel', 'BIO', 'TIF'] image characteristic
    Attributes:
        _cwd (str): current directory
        _filelist (str list): file list
        _image (mImage): current image file
        _curframe (int): current file number
    Commands:
        list:
        next:
        getfile(fileID):
        getframe(fileID, frameID):
    Todo:
        * documentation
    """

    def __init__(self, dirname, sort=False, debug=False, file_format='tif', **kwargs):
        """ imgfolder - object that handles the directory with all image files """

        # clean filename
        flist = glob.glob(dirname+'/*')
        for f in flist:
            newf = f.replace('__', '_')
            if f[0] == '_':
                newf = newf[1:]
            os.rename(f, newf)

        self._cwd = os.getcwd()
        if sort:
            self._filelist = sorted(glob.glob(dirname+'/*.'+file_format), key=getKey)
            self._tifidlist = sorted([getKey(x) for x in self._filelist])
        else:
            self._filelist = sorted(glob.glob(dirname+'/*.'+file_format))
            self._tifidlist = range(len(self._filelist))

        # database of files
        self._data = pd.DataFrame()
        self._data['id'] = self._tifidlist
        self._data['path'] = self._filelist
        self._data['frame'] = 0

        self._curidx = 0
        self._debug = debug
        self._image = ImageObject(self._data.at[self._curidx, 'path'], debug=self._debug, **kwargs)

        self._data['width'] = 0
        self._data['height'] = 0

        self._fileN = len(self._filelist)
        self._tdata = []

        self._window = [0, self._image._meta['width']]
        self._ranges = [0, self._image._meta['height']]

    def __repr__(self):
        """ representation """
        msg = ["... Path: %s" % self._cwd, "... # of files: {}".format(len(self._data))]
        return "\n".join(msg)

    def __getitem__(self, fileID):
        if ((fileID < 0) or (fileID >= self._fileN)):
            raise IndexError("frame range: %d to %d" % (0, self._fileN))
        self._curidx = fileID
        self._image = ImageObject(self._data.at[self._curidx, 'path'], debug=self._debug)
        self._data.at[fileID, 'width'] = self._image._meta['width']
        self._data.at[fileID, 'height'] = self._image._meta['height']

        return self._image

    def getfile(self, fileID=-1):
        if fileID is -1:
            fileID = self._curidx

        return self.__getitem__(fileID)

    def getframe(self, fileID=-1, frame=-1, types='pims'):
        self._image = self.__getitem__(fileID)

        return self._image.getframe(frame, types=types)

    # file selection
    def list(self):
        """ show file list in a directory """
        print('Current directory : {}'.format(self._cwd))
        print('Current idx: {}'.format(self._curidx))
        return self._data

    def search_filename(self, str_pattern):
        """ find file index from string """

        if self._fileN == 1:
            return -1
        tmp = self._data[self._data['path'].str.contains(str_pattern)]
        if len(tmp) > 0:
            return tmp

    def delete(self, fileidx=-1):
        """ delete file """
        if fileidx == -1:
            fileidx = self._curidx
        elif fileidx in range(self._fileN):
            pass
        else:
            print("... no file: [%i]" % (fileidx))

        try:
            print("... delete [%i] %s" % (fileidx, self._filelist[fileidx]))
            os.remove(self._filelist[fileidx])
            self._filelist.remove(self._filelist[fileidx])
            self._curidx -= 1
        except ValueError:
            print("... error")
            pass

    # plot
    def preview(self, update=False, ncol=-1, max_width=400, max_height=400):
        """ generate collective image of thumbnails
        ------------------------------------------------
        update: generate preview image from scratch
        ncol: number of columns in preview image
        max_width: size of each image
        """

        if 'cv2' not in dir(): import cv2

        tmp_filename = 'tmp_preview.png'

        # read from previous cache
        if (not update) and (ncol == -1) and os.path.exists(tmp_filename):
            preview_img = cv2.imread(tmp_filename, 0)    # read as gray scale
            ratio = preview_img.shape[0]/preview_img.shape[1]
            plt.figure(figsize=(16, int(16*ratio)))
            plt.imshow(preview_img, cmap='gray')
            plt.axis('off')
            return

        # prepare mean figures
        if ncol == -1: ncol = 6
        img_arr = []
        frameN_arr = []
        N = len(self._filelist)
        for i in tnrange(N):
            tif = ImageBase(self._filelist[i])
            im = _resize_image(tif.tmean(), max_width-2, max_height-2)

            # add new images
            img_arr.append(im)
            frameN_arr.append(tif._meta.N())

        # prepare output figure
        w, h = max_width, max_height
        nrow = int(np.ceil(N/ncol))
        res = np.ones((max_height * nrow, max_width * ncol), dtype=np.uint8) * 255
        if self._debug:
            print('... {} files, {} columns, {} rows, {} x {} pixels'.format(N, ncol, nrow, res.shape[0], res.shape[1]))

        # rescale and put in correct position
        for j in range(nrow):
            for i in range(ncol):
                tif_idx = i + j*ncol
                if tif_idx < self._fileN:
                    # get a new image dimension
                    ih, iw = img_arr[tif_idx].shape
                    # save in a new array
                    res[j*h + 1:j*h + ih + 1, i*w + 1:i*w + iw + 1] = img_arr[tif_idx]
                    cv2.putText(res, '%i (%i)' % (tif_idx, frameN_arr[tif_idx]), (i*w+2, j*h+27), 2, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
                    if self._debug: print('... [%i] %s' % (tif_idx, self._filelist[tif_idx]))

        plt.figure(figsize=(16, int(16.0*nrow/ncol)))
        plt.imshow(res, cmap='gray')
        plt.axis('off')

        if self._debug: print('... save to %s' % tmp_filename)
        cv2.imwrite(tmp_filename, res)
        plt.show()

    # track information
    def addtrack(self, t, tid=-1):
        if len(self._tdata) == 0:
            self._tdata = t
            self._tdata['particle'] = 0
        else:
            tidmax = self._tdata['particle'].max()
            tuni = np.unique(t['particle'].values)
            if tid != -1:
                a = np.isin(tuni, tid)
                if np.any(a):
                    tuni = [tid]
            for i in range(len(tuni)):
                tt = t[t['particle']==tuni[i]]
                tt['particle'] = tidmax+i+1
                self._tdata = pd.concat([self._tdata, tt])
            print('... %i tracks are added' % len(tuni))

    def deltrack(self, tid):
        if len(self._tdata) == 0:
            print('... no track data')
            exit

        tuni = np.unique(self._tdata['particle'].values)
        mask = np.isin(tuni, tid)
        if np.any(mask):
            self._tdata = self._tdata[self._tdata['particle'] != tid]

    def savetrack(self):
        if len(self._tdata) == 0:
            print('... no track data')
            exit
        fname = self._cwd.split('/')[-1]+'_trace.csv'
        self._tdata.to_csv(fname)
        print('... saved to %s' % fname)

    def loadtrack(self):
        fname = self._cwd.split('/')[-1]+'_trace.csv'
        if len(self._tdata) == 0:
            self._tdata = pd.DataFrame([])
        self._tdata = pd.read_csv(fname)
        print('... load from %s' % fname)

    def plottrack(self, windowsize=2.0):
        if len(self._tdata) == 0:
            print('... add tracks')
            exit

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ntrack = self._tdata['particle'].values.max() + 1
        width = windowsize
        height = windowsize
        Nx = np.ceil(np.sqrt(ntrack))+1

        unstacked = self._tdata.set_index(['particle', 'frame'])[['x', 'y']].unstack()
        for i, trajectory in unstacked.iterrows():
            x0 = width*(i % Nx)
            y0 = height*(np.ceil((i+1.0)/Nx))
            x = trajectory['x'].dropna().values
            y = trajectory['y'].dropna().values
            x = x - x[0] + x0
            y = y - y[0] + y0
            #print(x[0], y[0], i, x0, y0)
            ax.plot(x, y)
            plt.annotate(str(i), xy=(x0, y0), xytext=(x0-windowsize*0.3, y0+windowsize*0.3), arrowprops=dict(arrowstyle="-|>"), ha='center')

        plt.axis('equal')
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')
        ax.set_xlim([-windowsize*0.9, (Nx-0.5)*windowsize])
        ax.set_ylim([windowsize*0.8, (np.int(ntrack/Nx)+0.5)*windowsize])
        plt.tight_layout()
        plt.savefig(self._cwd.split('/')[-1]+"_traces_%i.pdf" % ntrack, dpi=300)
        plt.show()

    def caltrackmsd(self, m_to_p=0.0645, fps=2, plot=True):
        try:
            import trackpy as tp
        except ImportError:
            print('... install trackpy')
            exit

        if len(self._tdata) == 0:
            print('... add track data')
            exit

        em = tp.emsd(self._tdata, m_to_p, fps)
        a = tp.utils.fit_powerlaw(em, plot=False)
        tnumber = np.unique(self._tdata['particle'].values).max()+1

        if plot:
            fig = plt.figure(figsize=(6,4))
            x = pd.Series(em.index.values, index=em.index, dtype=np.float64)
            fits = x.apply(lambda x: a['A']*x**a['n'])
            plt.plot(em.index, em, 'o', label='exp. %i tracks' % tnumber)
            plt.plot(x, fits, label='y=%.4f x^{%.4f}'% (a['A'], a['n']))
            ax = plt.gca()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
            plt.legend(loc='best')
            plt.tight_layout()
            fname = self._cwd.split('/')[-1]+'_trace_msd.pdf'
            plt.savefig(fname, dpi=300)

            plt.show()

        return a

    # setting
    def setFit(self, on):
        if on:
            self._fitflag = True
            print('... Set Fit: On')
        else:
            self._fitflag = False
            print('... Set Fit: Off')

    def setSaveWindow(self, on):
        if on:
            self._windowflag = True
            print('... Set SaveWindow : On')
        else:
            self._windowflag = False
            print('... Set SaveWindow : Off')

    def setBaseline(self, base):
        self._baseline = base
        print('... Set baseline intensity: %f ' % base)

    def setColorMap(self, cmapname):
        """
        change color map
        possible map list:
            viridis, inferno, plasma, magma
            Greys, YlGn, bone, gray, summer
        """
        self._cmapname = cmapname
        self._image._meta['cmapname'] = self._cmapname
        print('... Set color map: '+self._cmapname)


def getKey(filename):
    file_text_name = filename.split('.')[:-1]

    file_id = file_text_name[-1].split('_')[-1]
    if file_id.isdigit():
        return int(file_id)

    file_id = file_text_name[-1].split('-')[-1]
    if file_id.isdigit():
        return int(file_id)

    file_id = file_text_name[-1]
    if file_id.isdigit():
        return int(file_id)

    return 10000


# vim:foldmethod=indent:foldlevel=0
