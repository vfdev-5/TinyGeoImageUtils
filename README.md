# Tiny Geographical Image Utils

[![Build Status](https://travis-ci.org/vfdev-5/TinyGeoImageUtils.svg?branch=master)](https://travis-ci.org/vfdev-5/TinyGeoImageUtils) 
[![Coverage Status](https://coveralls.io/repos/vfdev-5/TinyGeoImageUtils/badge.svg?branch=master&service=github&t=nhModO)](https://coveralls.io/github/vfdev-5/TinyGeoImageUtils?branch=master)


Reading and manipulation of geographical images. Based on [GDAL](http://www.gdal.org/)

## Installation:

### Dependencies:

#### Linux 
```
pip install numpy click
sudo apt-get install python3-gdal 
# or 
sudo apt-get install python-gdal
```

#### MacOSX

```
pip install numpy
brew install gdal
```

### Repository

```
pip install git+https://github.com/vfdev-5/TinyGeoImageUtils.git
```

## Basic tools:

### Python API

* `GeoImage` for reading geographical images 
* `GeoImageTiler` for tiled reading geographical images


####  Usage:

```
import numpy as np
import matplotlib.pyplot as plt
from gimg import GeoImage

gimg = GeoImage("path/to/image/file")
np_img = gimg.get_data([0, 0, 500, 500])

print np_img.shape, np_img.type
plt.imshow(np_img[:,:,0])
plt.show()
```

and 

```
import numpy as np
import matplotlib.pyplot as plt
from gimg import GeoImage
from gimg import GeoImageTilerConstSize

gimg = GeoImage("path/to/image/file")
tiles = GeoImageTilerConstSize(gimg, tile_size=(512, 512), min_overlapping=128)

for tile, x, y in tiles:
    print tile.shape, tile.type, x, y
```

See [example.ipynb](examples/examples.ipynb) for some basic examples of usage


### CLI 

* `tile_generator`

```
> 
