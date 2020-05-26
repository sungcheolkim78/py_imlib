"""
base_features.py - microscopy images class

"""

import pims
from tqdm import tqdm, tnrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import pandas as pd

from numba import njit
import pyfftw
import multiprocessing as mp

from .base_lines import ImageLine

__author__ = 'Sungcheol Kim <kimsung@us.ibm.com>'
__version__ = '1.0.0'


class ImageFeature(ImageLine):

    def __init__(self, objects, method='tifffile', debug=False, **kwargs):

        super().__init__(objects, method=method, debug=debug, **kwargs)
        """ TifFile class initialization """

        self._windowSize = 64
        self._overlap = 48
        self._searchArea = 72
        self._crop_window = [0, 0, self._meta['height'], self._meta['width']]
        self._dt = 0.01784
        self._piv_threshold = 1.3

        self._pivx = []
        self._pivy = []
        self._pivu = []
        self._pivv = []
        self._pivfilename = self._meta['basename']+'_pivdata'

        self._line_length = 20
        self._line_threshold = 20
        self._cansigma = 0

        self._shiftx = []
        self._shifty = []

    def crop(self, margin=30, box_arr=None, save=True, **kwargs):
        """ crop image for all frames """

        if self._debug: print('... width, height: %i, %i' % (self._width, self._height))

        # check given box_arr
        if box_arr is None:
            th = self.threshold(0, method='otsu', show=False, **kwargs)
            cnt_data = self.find_contour(0, threshold=th, show=self._debug, **kwargs)
            d = np.asarray(cnt_data)
            box_arr = d[:, 2:6]
        else:
            box_arr = [box_arr]

        # show image and box area
        fig = plt.figure(figsize=(11, 5))
        ax = fig.gca()
        self._showimage(self.mean(), frameNumber=False, ax=ax)
        for i, b in enumerate(box_arr):
            [x0, y0, x1, y1] = np.array(b) + np.array([-margin, -margin, margin, margin])

            [x0, x1] = np.clip([x0, x1], 0, self._height)
            [y0, y1] = np.clip([y0, y1], 0, self._width)

            x = [x0, x1, x1, x0, x0]
            y = [y0, y0, y1, y1, y0]
            ax.plot(x, y, '--', color='gray', alpha=0.8)
            ax.annotate(str(i), xy=(x0, y0), color='white')
            if self._debug: print("... [{}] ({}, {}, {}, {})".format(i, x0, y0, x1, y1))
        plt.show()

        # save cropped area
        a = input("Confirm? [Yes|No]")
        if a.upper() in ["YES", "Y"]:
            # only crop image data
            if (len(box_arr) == 1) and (save is not True):
                if self._debug: print('... only crop in image data')
                self._images = self._images[:, y0:y1+1, x0:x1+1]
                self._raw_images = self._raw_images[:, y0:y1+1, x0:x1+1]
            else:
                self.saveTif(box_arr=box_arr, margin=margin)
        else:
            return False

    def detect_drifts(self, frames=-1, min_size=200, show=True, save=False, full=True, **kwargs):
        """ detect drift by c.o.m """
        if 'cv2' not in sys.modules: import cv2

        if frames == -1:
            frames = range(self._frameN)

        # iterate through all frames
        xy = np.zeros((len(frames), 2))
        for i in tnrange(len(frames)):

            # find contour of cells
            th = self.threshold(frame=frames[i], show=False, **kwargs)
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
                shift_x = c[0] - xy[0, 0]
                shift_y = c[1] - xy[0, 1]
                M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
                img = self.getframe(i)
                shifted_img = cv2.warpAffine(img, M, (self._height, self._width), None, cv2.INTER_CUBIC, cv2.BORDER_WRAP)
                self._images[i, :, :] = shifted_img

        # save shift coordinate in panda dataframe
        result = pd.DataFrame(xy, index=frames, columns=['x', 'y'])

        if show:
            plt.imshow(self.getframe(frames[0]))
            plt.plot(xy[:, 0], xy[:, 1], 'white')
            plt.show()

        return result

    def detect_blob(self, frame=-1, res=5, **kwargs):
        """ detect blob using opencv or trackpy

        From trackpy locate commands
        -----------------------------
        minmass : float, optional
            The minimum integrated brightness. This is a crucial parameter for
            eliminating spurious features.
            Recommended minimum values are 100 for integer images and 1 for float
            images. Defaults to 0 (no filtering).
            .. warning:: The mass value is changed since v0.3.0
            .. warning:: The default behaviour of minmass has changed since v0.4.0
        maxsize : float
            maximum radius-of-gyration of brightness, default None
        separation : float or tuple
            Minimum separtion between features.
            Default is diameter + 1. May be a tuple, see diameter for details.
        noise_size : float or tuple
            Width of Gaussian blurring kernel, in pixels
            Default is 1. May be a tuple, see diameter for details.
        smoothing_size : float or tuple
            The size of the sides of the square kernel used in boxcar (rolling
            average) smoothing, in pixels
            Default is diameter. May be a tuple, making the kernel rectangular.
        threshold : float
            Clip bandpass result below this value. Thresholding is done on the
            already background-subtracted image.
            By default, 1 for integer images and 1/255 for float images.
        invert : boolean
            This will be deprecated. Use an appropriate PIMS pipeline to invert a
            Frame or FramesSequence.
            Set to True if features are darker than background. False by default.
        percentile : float
            Features must have a peak brighter than pixels in this
            percentile. This helps eliminate spurious peaks.
        topn : integer
            Return only the N brightest features above minmass.
            If None (default), return all features above minmass.
        preprocess : boolean
            Set to False to turn off bandpass preprocessing.
        max_iterations : integer
            max number of loops to refine the center of mass, default 10
        filter_before : boolean
            filter_before is no longer supported as it does not improve performance.
        filter_after : boolean
            This parameter has been deprecated: use minmass and maxsize.
        characterize : boolean
            Compute "extras": eccentricity, signal, ep. True by default.
        engine : {'auto', 'python', 'numba'}

        Returns
        -------
        DataFrame([x, y, mass, size, ecc, signal])
            where mass means total integrated brightness of the blob,
            size means the radius of gyration of its Gaussian-like profile,
            and ecc is its eccentricity (0 is circular).
        """

        U2 = self.getframe(frame)

        import trackpy as tp

        f = tp.locate(U2, res, **kwargs)
        fig = plt.figure(figsize=(11, 5))
        tp.annotate(f, np.hstack((U2, self.getframe(frame, orig=True, types=U2.dtype))))
        return f

    def detect_traces(self, frames=-1, res=5, margin=30, ids=None, **kwargs):
        """ detect particle tracks using trackpy and save as pt file

        From trackpy link command
        -------------------------
        f : DataFrame
            The DataFrame must include any number of column(s) for position and a
            column of frame numbers. By default, 'x' and 'y' are expected for
            position, and 'frame' is expected for frame number. See below for
            options to use custom column names.
        search_range : float or tuple
            the maximum distance features can move between frames,
            optionally per dimension
        pos_columns : list of str, optional
            Default is ['y', 'x'], or ['z', 'y', 'x'] when 'z' is present in f
        t_column : str, optional
            Default is 'frame'
        memory : integer, optional
            the maximum number of frames during which a feature can vanish,
            then reappear nearby, and be considered the same particle. 0 by default.
        predictor : function, optional
            Improve performance by guessing where a particle will be in
            the next frame.
            For examples of how this works, see the "predict" module.
        adaptive_stop : float, optional
            If not None, when encountering an oversize subnet, retry by progressively
            reducing search_range until the subnet is solvable. If search_range
            becomes <= adaptive_stop, give up and raise a SubnetOversizeException.
        adaptive_step : float, optional
            Reduce search_range by multiplying it by this factor.
        neighbor_strategy : {'KDTree', 'BTree'}
            algorithm used to identify nearby features. Default 'KDTree'.
        link_strategy : {'recursive', 'nonrecursive', 'numba', 'hybrid', 'drop', 'auto'}
            algorithm used to resolve subnetworks of nearby particles
            'auto' uses hybrid (numba+recursive) if available
            'drop' causes particles in subnetworks to go unlinked
        dist_func : function, optional
            a custom distance function that takes two 1D arrays of coordinates and
            returns a float. Must be used with the 'BTree' neighbor_strategy.
        to_eucl : function, optional
            function that transforms a N x ndim array of positions into coordinates
            in Euclidean space. Useful for instance to link by Euclidean distance
            starting from radial coordinates. If search_range is anisotropic, this
            parameter cannot be used.

        Returns
        -------
        DataFrame with added column 'particle' containing trajectory labels.
        The t_column (by default: 'frame') will be coerced to integer.
        """

        import trackpy as tp
        if frames == -1:
            frames = range(self._frameN)

        # using trackpy to select particles
        search_range = kwargs.pop('search_range', 10)
        memory = kwargs.pop('memory', 5)
        length = kwargs.pop('length', 5)
        psize = kwargs.pop('size', 0.0)
        ecc = kwargs.pop('ecc', 1.0)

        f = tp.batch(self._images[frames], res, **kwargs)
        t = tp.link_df(f, search_range, memory=memory)
        t1 = tp.filter_stubs(t, threshold=length)

        # select by size and eccentricity
        t2 = t1[((t1['size'] > psize) & (t1['ecc'] < ecc))]
        if (ids is not None) and isinstance(ids, list):
            t2 = t2.loc[t2['particle'].isin(ids)]

        pid = np.unique(t2['particle'])

        plt.clf()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca()
        xmin, xmax = int(t2['x'].min()), int(t2['x'].max())
        ymin, ymax = int(t2['y'].min()), int(t2['y'].max())
        xmin, xmax = np.clip([xmin-margin, xmax+margin], 0, self._height)
        ymin, ymax = np.clip([ymin-margin, ymax+margin], 0, self._width)

        for i, p in enumerate(pid):
            # plot individual traces

            # avoid crowd figure
            if i > 10:
                continue
            idx = np.where(t2['particle'] == p)[0]
            x = t2['x'].iloc[idx] - xmin
            y = t2['y'].iloc[idx] - ymin

            c = np.random.rand(3)*0.3 + [0.4, 0.1, 0.1]
            ax.plot(x, y, color=c, label='condensate %i (%i)' % (p, len(x)), alpha=0.9)
            ax.annotate('%i' % p, xy=(x.min(), y.min()), xytext=(x.min() - 15, y.min() - 15), va='bottom', ha='right', arrowprops=dict(arrowstyle="-|>", facecolor='white', edgecolor='white'), color='white')

        ax.imshow(self._raw_images[0, ymin:ymax+1, xmin:xmax+1])
        ax.axis('off')

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(self._fname[:-4] + '_trace.pdf', dpi=300)

        return t2

    def find_contour(self, frame=-1, threshold=None, show=True, min_size=30, max_size=-1, max_box=-1, plotmode='rbox'):
        """ find contour lines """
        if 'cv2' not in sys.modules: import cv2

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
            fig = plt.figure(figsize=(10, 5))
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
                    ax.plot(np.append(box[:, 0], box[0, 0]), np.append(box[:, 1], box[0, 1]), color=white_color, alpha=0.8)
                angle = angleR
                cx, cy = bx, by

            if show:
                ax.annotate(str(i), xy=(x, y), color='white')  # , fontsize='small')
                ax.annotate(str(i), xy=(x + img.shape[1], y), color='white')  # , fontsize='small')

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

    # detect frame shift
    def _find_frame_shift(self, img1, img2, show=False):
        from skimage.feature import register_translation
        res, error, diffphase = register_translation(img1, img2, 100)

        res = [-res[1], res[0]]
        if show:
            print('... maximum shift: {}'.format(res))
            print('... error: {}'.format(error))
            print('... diffphase: {}'.format(diffphase))
            plt.figure(figsize=(12, 6))
            plt.imshow(np.hstack((img1, img2)))

        return res

    def find_frame_shift(self, frame=-1, frame_delta=1, show=True):
        """ find drift from two sequel frames using phase correlation """

        if frame == -1:
            frame = self._curframe
        if frame + frame_delta > self._frameN - 1:
            raise KeyError('... out of frame range: {}, {}'.format(frame, frame + frame_delta))

        return self._find_frame_shift(self.getframe(frame), self.getframe(frame + frame_delta), show=show)

    def find_frame_shifts(self, show=True, method='fft'):
        """ calculate frame shift over all frames """
        shiftx = np.zeros(self._frameN-1)
        shifty = np.zeros(self._frameN-1)

        for i in trange(self._meta.N() - 1):
            if method == 'fft':
                res = find_shift(self.getframe(i), self.getframe(i+1))
            else:
                res = self._find_frame_shift(self.getframe(i), self.getframe(i+1), show=False)
            shiftx[i] = res[0]
            shifty[i] = res[1]

        self._shiftx = shiftx
        self._shifty = shifty

        if show:
            plt.plot(self._frames[:-1], shiftx, label='x')
            plt.plot(self._frames[:-1], shifty, label='y')
            plt.legend(loc='best')
            plt.xlabel('frames')
            plt.ylabel('x, y shifts [pixel]')

        return (shiftx.mean(), shifty.mean())

    def piv_frame(self, frame=-1, frame_delta=1, show=True, **kwargs):
        if frame == -1:
            frame = self._curframe
        img1 = self.getframe(frame, types='int32')
        img2 = self.getframe(frame + frame_delta, types='int32')

        return self._piv_frame(img1, img2, show=show, **kwargs)

    def _piv_frame(self, img1, img2, show=False, **kwargs):
        """
        calculate velocity using piv method on two frames
        """
        from openpiv.process import extended_search_area_piv, get_coordinates
        # from openpiv.scaling import uniform

        if self._debug:
            print('... [PIV] window size: {}'.format(self._windowSize))
            print('... [PIV] overlap: {}'.format(self._overlap))
            print('... [PIV] search area size: {}'.format(self._searchArea))
            print('... [PIV] threshold: {}'.format(self._piv_threshold))

        u, v, sig2noise = extended_search_area_piv(img1, img2, window_size=self._windowSize, overlap=self._overlap, dt=self._exposuretime, search_area_size=self._searchArea, sig2noise_method='peak2peak')
        self._pivx, self._pivy = get_coordinates(image_size=img1.shape, window_size=self._windowSize, overlap=self._overlap)
        #self._pivy = np.flipud(self._pivy)
        #self._pivx, self._pivy, u, v = uniform(self._pivx, self._pivy, u, v, scaling_factor=self._mpp)

        if show:
            from openpiv.validation import sig2noise_val
            from openpiv.filters import replace_outliers
            u, v, mask = sig2noise_val(u, v, sig2noise, threshold=self._piv_threshold)
            u, v = replace_outliers(u, v, method='localmean', max_iter=10, kernel_size=2)
            # show quiver plot
            plt.figure(figsize=(12, 6))
            plt.imshow(img1)
            plt.quiver(self._pivx, self._pivy, u, v, color='w', pivot='mid')
            plt.savefig(self._fname[:-4]+'_piv.png', dpi=100)

        if self._debug:
            print("... [PIV] mean velocity [um/sec]: ({:4.2f}, {:4.2f})".format(np.mean(u)*self._mpp, np.mean(v)*self._mpp))
            print("... [PIV] mean velocity [pixel/frame]: ({:4.2f}, {:4.2f})".format(np.mean(u)*self._exposuretime, np.mean(v)*self._exposuretime))

        return (u, v, sig2noise)

    def piv_frames(self, topn=-1, show=True):

        from openpiv.validation import sig2noise_val
        from openpiv.filters import replace_outliers

        frames = self._frames[::2]
        topn = np.min([len(frames), topn])
        frames = frames[:topn]

        (ut, vt, s2nt) = self.piv_frame(frame=0, show=False)
        for i in tqdm.tqdm(frames[1:]):
            (u, v, s2n) = self.piv_frame(frame=i, show=False)
            ut += u
            vt += v
            s2nt += s2n
            #print(np.max(u), np.min(u), u.size)
            #print(np.max(v), np.max(v), v.size)

        ut /= len(frames)
        vt /= len(frames)
        s2nt /= len(frames)

        ut, vt, mask = sig2noise_val(ut, vt, s2nt, threshold=self._piv_threshold)
        ut, vt = replace_outliers(ut, vt, method='localmean', max_iter=10, kernel_size=2)
        self._pivu = ut
        self._pivv = vt
        self._pivs2n = s2nt
        self.save_piv()

        if show:
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(121)
            ax1.imshow(self.mean())
            ax1.quiver(self._pivx, self._pivy, ut, vt, pivot='mid', color='w')

            ax2 = fig.add_subplot(122)
            n_u, bins, patches = ax2.hist(ut.flatten()*self._mpp, bins=20, normed=1, facecolor='blue', alpha=0.75, label='u')
            n_v, bins, patches = ax2.hist(vt.flatten()*self._mpp, bins=20, normed=1, facecolor='green', alpha=0.75, label='v')
            ax2.annotate(np.mean(ut)*self._mpp, xy=(np.mean(ut)*self._mpp, np.max(n_u)))
            ax2.annotate(np.mean(vt)*self._mpp, xy=(np.mean(vt)*self._mpp, np.max(n_v)))
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(self._fname[:-4]+'_piv_t.png', dpi=150)

        print("... frames: {}, {}".format(frames[0], frames[-1]))
        print("... mean velocity [um/sec]: ({:4.2f}, {:4.2f})".format(np.mean(ut)*self._umtopixel, np.mean(vt)*self._umtopixel))
        print("... mean velocity [pixel/frame]: ({:4.2f}, {:4.2f})".format(np.mean(ut)*self._dt, np.mean(vt)*self._dt))

    def get_piv_vel(self, window=[], verbose=True):
        if len(window) == 0:
            window = self._crop_window

        if len(self._pivu) == 0:
            print('... run find_piv_frames firt!')
            return

        idx_x = np.where(np.logical_and(window[0] <= self._pivx[0, :], self._pivx[0, :] <= window[2]))[0]
        idx_y = np.where(np.logical_and(window[1] <= self._pivy[:, 0], self._pivy[:, 0] <= window[3]))[0]
        print('... {} values are found!'.format(len(idx_x)*len(idx_y)))

        res_u = self._pivu[idx_y.min():idx_y.max(), idx_x.min():idx_x.max()]
        res_v = self._pivv[idx_y.min():idx_y.max(), idx_x.min():idx_x.max()]

        mean_u = np.mean(res_u)*self._umtopixel
        mean_v = np.mean(res_v)*self._umtopixel
        print("... mean velocity [um/sec]: ({:4.2f}, {:4.2f})".format(mean_u, mean_v))
        print("... mean velocity [pixel/frame]: ({:4.2f}, {:4.2f})".format(np.mean(res_u)*self._dt, np.mean(res_v)*self._dt))

        if verbose:
            res_x = self._pivx[idx_y.min():idx_y.max(), idx_x.min():idx_x.max()]
            res_y = self._pivy[idx_y.min():idx_y.max(), idx_x.min():idx_x.max()]

            fig = plt.figure(figsize=(10, 4))

            ax = fig.add_subplot(121)
            ax.imshow(self.getMean())
            ax.quiver(self._pivx, self._pivy, self._pivu, self._pivv, pivot='mid', color='w')
            ax.quiver(res_x, res_y, res_u, res_v, pivot='mid', color='y')

            ax = fig.add_subplot(122)
            n_u, bins, patches = ax.hist(res_u.flatten()*self._umtopixel, bins=20, normed=1, facecolor='blue', alpha=0.75, label='u')
            n_v, bins, patches = ax.hist(res_v.flatten()*self._umtopixel, bins=20, normed=1, facecolor='green', alpha=0.75, label='v')
            plt.annotate('{:4.2f} [um/sec]'.format(mean_u), xy=(mean_u, n_u.max()/2.0))
            plt.annotate('{:4.2f} [um/sec]'.format(mean_v), xy=(mean_v, n_v.max()/2.0))

            plt.legend(loc='best')
            plt.tight_layout()
            filename = self._filename[:-4]+'_piv_w%d_%d_%d_%d.png' % (window[0], window[1], window[2], window[3])
            plt.savefig(filename, dpi=150)
            plt.show()

    def load_piv(self):
        if os.path.exists(self._pivfilename+'.npy'):
            a = np.load(self._pivfilename+'.npy')
            self._pivx, self._pivy, self._pivu, self._pivv = np.hsplit(a, 4)
        else:
            print('... %s is not exist' % self._filename)
            return

    def save_piv(self):
        a = np.hstack((self._pivx, self._pivy, self._pivu, self._pivv))
        np.save(self._pivfilename, a)


def shift2(arr):
    """ shift 2d array with half of its dimension """

    d = int(arr.shape[0]/2), int(arr.shape[1]/2)
    cd = arr.shape[0] - d[0], arr.shape[1] - d[1]
    tmp = np.zeros_like(arr)

    tmp[:d[0], :d[1]] = cc[-d[0]:, -d[1]:]
    tmp[:d[0], d[1]:] = cc[-d[0]:, :cd[1]]
    tmp[d[0]:, :d[1]] = cc[:cd[0], -d[1]:]
    tmp[d[0]:, d[1]:] = cc[:cd[0], :cd[1]]

    return tmp


#@njit(fastmath=True)
def find_shift(image0, image1):
    """ find drift velocity using fft and correlation """
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.config.NUM_THREADS = mp.cpu_count()

    # convert data type
    im0 = pyfftw.empty_aligned(image0.shape, dtype='complex128')
    im1 = pyfftw.empty_aligned(image1.shape, dtype='complex128')
    im0 = image0
    im1 = image1

    # run fast fourier transformation
    cc = fft.ifft2(fft.fft2(image0)*np.conj(fft.fft2(image1)))

    #if show:
    #    plt.imshow(np.abs(shift2(cc)))

    coord = np.unravel_index(cc.argmax(), cc.shape)

    y = cc.shape[0] - coord[0] if coord[0] > cc.shape[0]/2 else coord[0]
    x = cc.shape[1] - coord[1] if coord[1] > cc.shape[1]/2 else coord[1]

    #return [coord[0] - d[0], coord[1] - d[1]]
    return [x, -y]

# vim:foldmethod=indent:foldlevel=0
