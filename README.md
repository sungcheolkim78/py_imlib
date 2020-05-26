# imlib

Image library for internal usage of Nanobiotechnology team

## install

There are preliminary packages for imlib.

```bash
pip3 install tifffile
pip3 install tqdm
pip3 install numpy
pip3 install scipy
pip3 install pandas
pip3 install pims

brew install opencv4

pip install "https://github.ibm.com/kimsung/imlib.git"
```

`opencv` is a package hard to install. In macOS, use [homebrew]() and in Ubuntu, use apt-get to install opencv. (Currently - 20190325, opencv4 is a mainstream version.)

## Quick start

For best practice, I recommend to use jupyter notebook. In IBM network, you can log in `nanobio.watson.ibm.com` and start jupyter notebook session.

To analysis tiff images in a single directory, move to the directory of interest.

```python
import imlib

imgfolder = imlib.ImgFolder('.')
imgfolder.preview()
```

These commands create `ImgFolder` object and it automatically include all images inside a current directory. `preview()` method creates thumbnail images of all files.

```python
tif = imgfolder.get(1)
tif.mean()
```

You can analyze an individual file by using `get()` method and it returns `ImgBase` object which has basic functions for analysis. `mean()` method gives the averaged image over all frames.
