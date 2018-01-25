# Tiny Geographical Image Utils

Reading and manipulation of geographical images. Based on [GDAL](http://www.gdal.org/)

## Installation:

### Dependencies:

#### Linux
```
pip install numpy
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
pip install git+git://github.com/vfdev-5/TinyGeoImageUtils
```

## Basic tools:

* `GeoImage` for reading geographical images 

* `GeoImageTiler` for tiled reading geographical images


## Basic usage:

```
import numpy as np
import matplotlib.pyplot as plt
from gimg.GeoImage import GeoImage

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
from gimg.GeoImage import GeoImage
from gimg.GeoImage import GeoImageTilerConstSize

gimg = GeoImage("path/to/image/file")
tiles = GeoImageTilerConstSize(gimg, tile_size=(512, 512), min_overlapping=128)

for tile, x, y in tiles:
    print tile.shape, tile.type, x, y
```

See [example.ipynb](examples/examples.ipynb) for some basic examples of usage
