"""
meta.py - object for storing meta information

date: 20191023 - separated from base.py
"""

import os
import pickle

class Meta(object):
    """ class for meta information """

    def __init__(self, fname):
        """ storage for all meta information in image analysis """

        basename = os.path.splitext(fname)[0]
        basename = basename.split('/')[-1]
        self._dict = dict(
                frameN=1, width=512, height=512, 
                nd2_m=0, nd2_c=0, 
                fname=fname, cwd=os.path.dirname(os.path.abspath(fname)),
                ext=os.path.splitext(fname)[-1][1:], 
                basename=basename, 
                exposuretime=0.0, magnification='10x', 
                mpp=1.0, ppm=1.0,
                duration=0.0, 
                cmapname='viridis',
                openMethod='unknown', 
                wall1=0, wall2=512,
                range1=0, range2=512,
                channelWidth=512, 
                pillarRatio=10, u=100, v=0, p=0, psize=50,
                mangle=0, fangle=0, D=0, Pe=0,
                filterProtocol=''
                )

        self._source = []
        self._zrange = [0, 0]
        self._metafname = os.path.join(self._dict['cwd'], 
                './.{}_meta.csv'.format(self._dict['basename']))
        if os.path.isfile(self._metafname): self.load()

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, item):
        self._dict[key] = item

    def __repr__(self):
        msg = ''
        msg += '... File Name: %s\n' % self['fname']
        msg += '... File Basename: %s\n' % self['basename']
        msg += '... File Extension: %s\n' % self['ext']
        msg += '... File Path: {}\n'.format(self['cwd'])
        msg += '... Frames: %i\n' % self['frameN']
        msg += '... Frame Size: (%i, %i)\n' % (self['width'], self['height'])
        msg += '... Range: (%f, %f)\n' % (self._zrange[0], self._zrange[1])
        msg += '... Colormap: %s\n' % self['cmapname']
        msg += '... Exposure Time: %f\n' % self['exposuretime']
        msg += '... Duration: {}\n'.format(self['duration'])
        msg += '... Magnification: {}\n'.format(self['magnification'])
        msg += '... Micrometer per Pixel: {:.4f} [um/pixel]\n'.format(self['mpp'])

        return msg
        #return repr(self._dict)

    def __len__(self):
        return len(self._dict)

    def update_dim(self, n, w, h):
        self['frameN'] = n
        self['width'] = w
        self['height'] = h
        self.save()

    def update_exp(self, exp):
        self['exposuretime'] = exp
        self['duration'] = self['exposuretime'] * self['frameN']
        self.save()

    def update_mag(self, magnification, ccd_length=16.0):
        self['magnification'] = magnification
        try:
            mag = float(magnification[:-1])
            self['mpp'] = ccd_length /mag  # andor iXon camera pixel size : 16 um
            self['ppm'] = mag / ccd_length
            self.save()
        except:
            print('... magification: 100x, 63x, 40x, ... : {}'.format(magnification))

    def update_wall(self, wallinfo):
        if sum(wallinfo) > 0:
            self['wall1'] = wallinfo[0]
            self['wall2'] = wallinfo[1]
            self['range1'] = wallinfo[2]
            self['range2'] = wallinfo[3]
            self['channelWidth'] = wallinfo[1] - wallinfo[0]
        self['arrayWidth'] = self['channelWidth'] * self['mpp']            # [um]
        self['arrayLength'] = self['pillarRatio'] * self['arrayWidth']     # [um]
        self.save()

    def N(self):
        return int(self['frameN'])

    def save(self):
        pickle.dump(self._dict, open(self._metafname, 'wb'))

    def load(self):
        self._dict = pickle.load(open(self._metafname, 'rb'))
                

# vim:foldmethod=indent:foldlevel=0