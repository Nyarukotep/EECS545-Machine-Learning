import urllib.request, urllib.error, urllib.parse
import os, tempfile

import numpy as np
import imageio


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imageio.imread(fname)
        #If there is an error from "os.remove(fname)". Please try commenting
        #out this line and running the code. The code then won't delete the
        #downloaded file for you, but you can delete it manually later.
        #os.remove(fname)
        print('Successfully get image from', url)
        return img
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)
        raise
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
        raise
