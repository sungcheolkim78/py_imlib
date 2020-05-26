"""
base.py - base images class

date: 20180712 - version 1.0.0 - new class with sub-class idea
date: 20191022 - version 1.1.0 - clean up and remove cache mechanism and add Meta class
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

import pims
import tifffile

from .meta import Meta

__author__ = 'Sungcheol Kim <kimsung@us.ibm.com>'
__docformat__ = 'restructuredtext en'
__version__ = '1.1.0'

__all__ = ('ImageBase', '_rescale_to_dtype', '_as_grey', '_resize_image')


class ImageBase(object):
    """ basic image object with minimum methods """

    def __init__(self, objects, method='tifffile', debug=False, **kwargs):
        """ initialize ImageBase class """
        # set debug flag
        self._debug = debug

        # file information
        if isinstance(objects, np.ndarray) or isinstance(objects, pims.Frame):
            if objects.ndim == 3:
                self._meta = Meta('temp.tif')
                self._meta.update_dim(objects.shape[0], objects.shape[1], objects.shape[2])
                self._images = pims.Frame(objects)
            elif self._raw_images.ndim == 2:
                self._meta = Meta('temp.png')
                self._meta.update_dim(1, objects.shape[0], objects.shape[1])
                self._images = pims.Frame(objects)
        elif isinstance(objects, ImageBase):
            self._meta = ImageBase._meta
            self._images = objects._images
        elif isinstance(objects, str):
            self._meta, self._images = self.open(objects, method=method)

        self._curframe = 0
        self._single = True if self._meta['frameN'] == 1 else False

        # cache 
        self._raw_images = self._images.copy()  
        self._imgMean = []
        self._imgMax = []
        self._imgMin = []
        self._imgMedian = []

    def open(self, filename, method='tifffile', nd2_m=0, nd2_c=1):
        """ read file """
        # set filename or find filename
        if not os.path.isfile(filename):
            raise IOError('... no file: {}'.format(filename))

        ext = filename.split('.')[-1]

        # using bioformat library
        if (method == 'bioformat'):
            return _open_bioformat(filename, debug=self._debug)
        # nd2 format
        elif ext == 'nd2':
            return _open_nd2(filename, nd2_m, nd2_c, debug=self._debug)
        # tif format with tifffile library - fastest method
        elif (ext == 'tif') and (method == 'tifffile'):
            return _open_tif(filename, debug=self._debug)
        elif (ext == 'sif'):
            return _open_sif(filename, debug=self._debug)
        # other format
        else:
            return _open_pims(filename, debug=self._debug)

    def __repr__(self):
        """ representation of TifFile """

        return self._meta.__repr__()

    # data conversion
    def asframe(self):
        """ return pims Frame object """

        return pims.Frame(self._images)

    def asarray(self):
        """ return numpy array for frames """

        return self._images

    def check_frame(func):
        @wraps(func)
        def frame_func(self, **kwargs):
            i = kwargs.pop('frame', -1)
            if (i < 0) or (i > self._meta.N()-1): i = self._curframe
            return func(self, frame=i, **kwargs)
        return frame_func

    def __getitem__(self, frame):
        if self._single:
            return self._images
        else:
            return self._images[frame, :, :]

    @check_frame
    def getframe(self, frame=-1, dtypes='pims'):
        """ get frame """

        if dtypes in ['uint8', 'uint16', 'int32', 'float', 'orig']:
            if self._single:
                self._images = _rescale_to_dtype(self._raw_images, dtypes)
            else:
                img = self._raw_images[frame, :, :]
                self._images[frame, :, :] = _rescale_to_dtype(img, dtypes)

        img = self[frame]

        if self._debug: print('... getframe [{}]: {}, {}, {}, {}'.format(frame, img.min(), img.mean(), img.max(), img.dtype))

        return pims.Frame(img)

    @check_frame
    def reverse(self, frame=-1):
        """ reverse frame back to original """

        if self._single:
            self._images = self._raw_images
        else:
            self._images[frame, :, :] = self._raw_images[frame, :, :]

    def tmean(self, dtypes='orig'):
        """ calculate frame mean image """
        return _rescale_to_dtype(self._images.mean(axis=0), dtypes)

    def tmax(self, dtypes='orig'):
        """ calculate frame maximum image """
        return _rescale_to_dtype(self._images.max(axis=0), dtypes)

    def tmin(self, dtypes='orig'):
        """ calculate frame minimum image """
        return _rescale_to_dtype(self._images.min(axis=0), dtypes)

    def tmedian(self, dtypes='orig'):
        """ calculate frame minimum image """
        return _rescale_to_dtype(np.median(self._images, axis=0))

    # plot functions
    def show(self, frame=-1, dtypes='orig', autorange=True, **kwargs):
        self._show(self.getframe(frame=frame, dtypes=dtypes), autorange=autorange, **kwargs)

    def _show(self, img, autorange=True, **kwargs):
        """ show image and histogram together for analysis """
        plt.clf()
        fig = plt.figure(figsize=(11, 5))

        ax1 = fig.add_subplot(121)
        self._showimage(img, simple=False, autorange=autorange, ax=ax1)

        ax2 = plt.axes([0.62, 0.32, 0.35, 0.55])
        msg = self._hist(img, ax=ax2, autorange=autorange, **kwargs)
        ax2.text(0.1, -0.4, msg, ha='left', transform=ax2.transAxes)

    def _showimage(self, img, simple=False, autorange=True, \
            frameNumber=True, wall=False, ax=None, **kwargs):
        """ base plot function with axis  and colorbar """

        zrange = self._meta._zrange
        if autorange: zrange = [img.min(), img.max()]

        height, width = img.shape

        if ax is None:
            plt.clf()
            fig = plt.figure(figsize=(10, 5))
            ax = fig.gca()

        self._im = ax.imshow(img, clim=zrange, cmap=self._meta['cmapname'], origin='image', **kwargs)

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        if wall:
            wallinfo = [self._meta['wall1'], self._meta['wall2'], self._meta['range1'], self._meta['range2']]

            ax.vlines(wallinfo[0], 0, height, color='w', linestyles='dashed')
            ax.annotate('a {}'.format(wallinfo[0]), xy=(wallinfo[0]+3, 0), va='top', ha='left', color='w')

            ax.vlines(wallinfo[1], 0, height, color='w', linestyles='dashed')
            ax.annotate('b {}'.format(wallinfo[1]), xy=(wallinfo[1]-3, height), va='bottom', ha='right', color='w')

            ax.hlines(wallinfo[2], wallinfo[0], wallinfo[1], color='w', linestyles='dashed')
            ax.annotate('c {}'.format(wallinfo[2]), xy=(wallinfo[0]+3, wallinfo[2]+3), va='top', color='w')

            ax.hlines(wallinfo[3], wallinfo[0], wallinfo[1], color='w', linestyles='dashed')
            ax.annotate('d {}'.format(wallinfo[3]), xy=(wallinfo[0]+3, wallinfo[3]-3), va='bottom', color='w')

        if not simple:
            if frameNumber:
                ax.annotate('%d/%d' % (self._curframe, self._meta.N()), xy=(0, 0))

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            ax.axis('on')
            ax.axis((0, img.shape[1], img.shape[0], 0))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            plt.colorbar(self._im, cax=cax)
            ax.set_xlabel('x (pixels)')
            ax.set_ylabel('y (pixels)')

    def showFrame(self, frame=-1, simple=False, **kwargs):
        """ plot frame with axis """

        img_f = self.getframe(frame=frame)
        self._im = self._showimage(img_f, simple=simple, **kwargs)

    def showMean(self, simple=False):
        self._im = self._showimage(self.tmean(), simple=simple, frameNumber=False)

    def showMax(self, simple=False):
        self._im = self._showimage(self.tmax(), simple=simple, frameNumber=False)

    def showMin(self, simple=False):
        self._im = self._showimage(self.tmin(), simple=simple, frameNumber=False)

    def LUT(self, cmapname=None):
        """ change/view colormap """
        LUT_table = ['viridis', 'plasma', 'inferno', 'magma', 'gray', 'bone', 'pink', 'summer', 'afmhot', 'rainbow', 'jet', 'ocean', 'gnuplot', 'gnuplot2']

        if cmapname is None:
            print('... current LUT: {}'.format(self._cmapname))
        elif cmapname == 'next':
            if self._meta['cmapname'] in LUT_table:
                idx = LUT_table.index(self._meta['cmapname'])
                self._meta['cmapname'] = LUT_table[idx+1] if idx < len(LUT_table) - 1 else LUT_table[0]
                print('... set LUT: {}'.format(self._meta['cmapname']))
        elif cmapname == 'prev':
            if self._cmapname in LUT_table:
                idx = LUT_table.index(self._cmapname)
                self._cmapname = LUT_table[idx-1] if idx > 0 else LUT_table[-1]
                print('... set LUT: {}'.format(self._cmapname))
        elif cmapname == 'help':
            print('... Look Up Table (LUT) list: {}'.format(', '.join(LUT_table)))
        elif isinstance(cmapname, str):
            self._meta['cmapname'] = cmapname
            print('... set LUT: {}'.format(self._meta['cmapname']))

    # histogram
    def histFrame(self, frame=-1, dtypes='orig', normal=False, bins=256):
        return self._hist(self.getframe(frame=frame, dtypes=dtypes), frame=frame, normal=normal, bins=bins)

    def histMax(self, dtypes='orig', normal=False, bins=256):
        return self._hist(self.tmax(dtypes=dtypes), frame='max', normal=normal, bins=bins)

    def histMin(self, dtypes='orig', normal=False, bins=256):
        return self._hist(self.tmin(dtypes=dtypes), frame='min', normal=normal, bins=bins)

    def histMean(self, dtypes='float', normal=False, bins=256):
        return self._hist(self.tmean(dtypes=dtypes), frame='mean', normal=normal, bins=bins)

    def _hist(self, img, frame=-1, normal=False, bins=256, autorange=True, ax=None):
        """ plot histogram """

        if self._debug: print('... dtype: {}'.format(img.dtype))

        h, x = np.histogram(img.ravel(), bins=bins)

        if ax is None:
            ax = plt.gcf().gca()

        if autorange:
            ax.set_xlim(x.min(), x.max())
        else:
            ax.set_xlim(self._zrange[0], self._zrange[1])
        ax.set_xlabel('intensity')
        labels = frame if isinstance(frame, str) else str(frame)
        if normal:
            h_sum = h.sum()
            self._im = ax.plot(x[:-1], h/h_sum, label=labels)
            ax.set_ylabel('Normalized count')
            ax.set_ylim(h.min()/h_sum, h.max()/h_sum)
        else:
            self._im = ax.plot(x[:-1], h, label=labels)
            ax.set_ylabel('count')
            ax.set_ylim(h.min(), h.max())
        ax.legend()

        # TODO make alignment correct
        res_msg = 'Mean: {:<12.4f}     Min: {:<12.4f} \nStd:    {:<12.4f}   Max: {:<12.4f} \nbins:   {:<12d}   bin width: {:<12.2f}'.format(img.mean(), img.min(), img.std(), img.max(), bins, x[1] - x[0])
        return res_msg

    # save
    def _save(self, img, appendix='', prefix='', savename=None, format='tif'):
        """ save frame as tif with a name + appendix """
        if savename is None:
            savename = prefix + self._meta['fname'][:-4] + appendix + '.' + format
        if format == 'tif':
            tifffile.imsave(savename, img, dtype=img.dtype)
        else:
            from matplotlib import image
            image.imsave(savename, img)

    def saveMean(self, update=False, dtypes='orig', prefix='.', format='tif'):
        self._save(self.tmean(dtypes=dtypes), appendix='_mean', prefix='.', format=format)

    def saveMax(self, update=False, dtypes='orig', prefix='.', format='tif'):
        self._save(self.tmax(dtypes=dtypes), appendix='_max', prefix='.', format=format)

    def saveMin(self, update=False, dtypes='orig', prefix='.', format='tif'):
        self._save(self.tmin(dtypes=dtypes), appendix='_min', prefix='.', format=format)

    def saveTif(self, appendix=-1, box_arr=None, margin=0):
        """ save as tif format for saving data """

        if appendix == -1:
            appendix = ''
        if self._meta['ext'] == 'nd2':
            nd_code = '-m%ic%i-' % (self._nd2_m, self._nd2_c)
        else:
            nd_code = ''

        # check box information
        if box_arr is None:
            savefname = self._meta['fname'][:-4] + nd_code + appendix + '.tif'

            if os.path.exists(savefname):
                a = input('... overwrite %s [Yes/No]?' % savefname)
                if a.upper() in ['NO', 'N']:
                    return False

            print('... save to %s' % savefname)
            tifffile.imsave(savefname, self._images)

        # with box selections
        elif len(box_arr) > 0:
            # for each boxes
            for (i, box) in enumerate(box_arr):
                savefname = self._meta['fname'][:-4] + nd_code + 'crop_%i' % i + '.tif'

                if os.path.exists(savefname):
                    a = input('... overwrite %s [Yes/No]?' % savefname)
                    if a.upper() in ['NO', 'N']:
                        continue

                [x0, y0, x1, y1] = np.array(box) + np.array([-margin, -margin, margin, margin])
                # adjust values not to overbound
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(self._meta['height'], x1), min(self._meta['width'], y1)

                print('... save to %s' % savefname)
                if self._frameN > 1:
                    tifffile.imsave(savefname, self._images[:, y0:y1 + 1, x0:x1 + 1])
                else:
                    tifffile.imsave(savefname, self._images[y0:y1 + 1, x0:x1 + 1])

    def saveMovie(self, zoomfactor=1.0, savename='None', appendix='', update=False, fps=25):
        """
        save tif file as mp4 using moviepy library. The duration of movie file
        is determined as 3 times of realtime video.

        Input:
        zoomfactor = 0.5 (default), save image using resampling
        savename = default format [tif file name]_z[zoomefactor].mp4

        Return:
        VidoeClip object in moviepy library
        """

        if ('moviepy' not in dir()):
            from moviepy.editor import VideoClip
        if ('cv2' not in dir()):
            import cv2

        if (savename == 'None'):
            savename = '{}_z{:.1f}_{}.mp4'.format(self._meta['fname'][:-4], zoomfactor, appendix)
            if not update:
                if os.path.exists(savename):
                    if self._debug: print('... movie file already exists: %s' % savename)
                    return False

        if self._single:
            if self._debug: print('... not movie file')
            return False

        cmap = plt.get_cmap(self._meta['cmapname'])

        def make_frame(t):
            #self._curframe = int(t * (self._frameN - 1) / (self._duration * 3.0))
            self._curframe = int(t * fps)
            img0 = self.getframe(frame=self._curframe, dtypes='uint8')
            if zoomfactor != 1.0:
                img0 = cv2.resize(img0, None, fx=zoomfactor, fy=zoomfactor, interpolation=cv2.INTER_CUBIC)
            img = np.delete(cmap(img0), 3, 2)
            #return img
            return (img * 255.0).astype('uint8')

        #animation = VideoClip(make_frame, duration=self._duration * 3.0)
        animation = VideoClip(make_frame, duration=float(self._meta.N()/fps))

        animation.write_videofile(savename, fps=fps, codec='libx264', \
                threads=8, audio=False, preset='medium', verbose=False)
        self._animation = animation

        print("""To play movie file in jupyter notebook:
            from IPython.display import Video
            Video("{}")""".format(savename))


def _open_tif(filename, debug=False):
    """ open tif file using tifffile library """

    if debug: print('... read %s with tifffile library' % filename)

    meta = Meta(filename)
    meta['openMethod'] = 'tifffile'

    with tifffile.TiffFile(filename) as tif:
        imgarray = tif.asarray()
        try:
            meta._source = tif.andor_metadata
            meta.update_dim(tif.andor_metadata['Frames'], 
                tif.andor_metadata['ChipSizeX'], 
                tif.andor_metadata['ChipSizeY'])
            meta.update_exp(tif.andor_metadata['AcquisitionCycleTime'])
        except:
            if debug: print('... no meta data')

            if imgarray.ndim == 2:
                meta.update_dim(1, imgarray.shape[0], imgarray.shape[1])
            elif imgarray.ndim == 3:
                meta.update_dim(imgarray.shape[0], imgarray.shape[1], imgarray.shape[2])

    return meta, imgarray


def _open_sif(filename, debug=False):
    """ open sif file using sif_reader """

    if 'sif_reader' not in dir(): import sif_reader

    if debug: print('... read %s with sif_reader library' % filename)

    meta = Meta(fname)
    meta['openMethod'] = 'sif_reader'

    with sif_reader.np_open(filename) as f:
        meta.update_dim(f[1]['NumberOfFrames'], f[1]['DetectorDimensions'][0], f[1]['DetectorDimensions'][1])
        meta.update_exp(f[1]['ExposureTime'])
        meta._source = f[1]
        imgarray = f[0][0] if meta.N() == 1 else f[0]

    return meta, imgarray


def _open_nd2(filename, nd2_m, nd2_c, debug=False):
    """ open nd2 file using pims-nd library """

    if debug: print('... read %s with nd2_reader library' % filename)

    meta = Meta(filename)
    meta['openMethod'] = 'pims-nd'
    
    with pims.ND2_Reader(filename) as tif:
        if 't' in tif.sizes:
            tif.bundle_axes = 'tyx'
        if 'm' in tif.sizes:
            if tif.sizes['m'] > 1:
                tif.iter_axes = 'm'
                print('... multistack image %i of %i selected' % (nd2_m + 1, tif.sizes['m']))
        if 'c' in tif.sizes:
            tif.default_coords['c'] = nd2_c
            print('... multichannel image %i of %i selected' % (nd2_c + 1, tif.sizes['c']))

        imgarray = np.array(tif[nd2_m])
        meta.update_dim(tif.sizes['t'], tif.sizes['x'], tif.sizes['y'])
        meta.update_exp(tif[0].metadata['t_ms'])
        meta._mpp = tif.metadata['calibration_um']

    return meta, imgarray


def _open_bioformat(filename, debug=False):
    """ open file using bioformat library """

    if debug: print('... read %s with bioformat library' % filename)

    meta = Meta(filename)
    meta['openMethod'] = 'bioformat'

    with pims.Bioformats(filename) as tif:
        imgarray = np.array(tif[:])
        try:
            meta._source = tif.metadata
            meta.update_exp(tif.metadata.PlaneExposureTime(0, 0))
            meta['mpp'] = tif.metadata.PixelsPhysicalSizeX(0)
            frameN = tif.sizes['t']
        except:
            frameN = 1
        meta.update_dim(frameN, tif.metadata.PixelsSizeX(0), tif.metadata.PixelsSizeY(0))

    return meta, imgarray


def _open_pims(filename, debug=False):
    """ open file using pims library """

    meta = Meta(filename)
    meta['openMethod'] = 'pims'

    with pims.open(filename) as tif:
        if len(tif.frame_shape) == 2:    # grey image
            if debug: print('... read %s with pims library - black/white' % filename)
            imgarray = np.array(tif[:])

            # in case of single image
            if len(tif) == 0:
                imgarray = imgarray[0]
        else:   # color image
            if debug: print('... read %s with pims library - converted to b/w' % filename)
            imgarray = _as_grey(tif)

        meta.update_dim(len(tif), tif.frame_shape[0], tif.frame_shape[1])

    return meta, imgarray


def _rescale_to_dtype(oimg, dtypes):
    """ convert image data type """

    if 'rescale_intensity' not in dir(): from skimage.exposure import rescale_intensity

    olim = (oimg.min(), oimg.max())

    if dtypes in ['uint8', 'uint16', 'int16']:
        return rescale_intensity(oimg, in_range=olim, out_range=dtypes).astype(dtypes)
    elif dtypes == 'float':
        #oimg = oimg.astype('float64')
        return rescale_intensity(oimg, in_range=olim, out_range=(0.0, 1.0)).astype('float')
    else:
        return oimg


def _as_grey(frame):
    array = np.array(frame[:])
    if array.shape[0] == 1:
        array = array[0, ...]
    return np.dot(array[..., :3], [0.2125, 0.7154, 0.0721])


def _resize_image(img, max_width, max_height):
    """ resize image into image of size width and height """

    if 'cv2' not in dir(): import cv2

    if len(img.shape) == 3:
        raise ValueError("img should be gray image")

    o_height, o_width = img.shape
    if o_height > o_width:
        scale = max_height/o_height
    else:
        scale = max_width/o_width

    img = _rescale_to_dtype(img, 'uint8')
    if scale < 1.0:
        im = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        im = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return im

# vim:foldmethod=indent:foldlevel=0
