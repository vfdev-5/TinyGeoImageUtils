# Tiny Geographical Image Utils

[![Build Status](https://travis-ci.org/vfdev-5/TinyGeoImageUtils.svg?branch=master)](https://travis-ci.org/vfdev-5/TinyGeoImageUtils) 
[![Coverage Status](https://coveralls.io/repos/github/vfdev-5/TinyGeoImageUtils/badge.svg)](https://coveralls.io/github/vfdev-5/TinyGeoImageUtils)

Reading and manipulation of geographical images. Based on [GDAL](http://www.gdal.org/)

## Installation:

### Dependencies:

#### Linux 
```
pip install numpy click
sudo apt-get install libgdal-dev
pip install gdal
```

#### MacOSX

```
pip install numpy
brew install gdal2
pip install gdal
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

```python
import numpy as np
import matplotlib.pyplot as plt
from gimg import GeoImage

gimg = GeoImage("path/to/image/file")
data = gimg.get_data([0, 0, 500, 500])

print(data.shape, data.type)
plt.imshow(data[:, :, 0])
plt.show()
```

and 

```python
import numpy as np
import matplotlib.pyplot as plt
from gimg import GeoImage
from gimg import GeoImageTilerConstSize

gimg = GeoImage("path/to/image/file")
tiles = GeoImageTilerConstSize(gimg, tile_size=(512, 512), min_overlapping=128)

for tile, x, y in tiles:
    print(tile.shape, tile.type, x, y)
```

See other examples: 
- [example.ipynb](examples/examples.ipynb)
- [create images](examples/create_images.py)


### CLI 

#### `tile_generator` 

Application to write tiles from input single image or a folder of images. 

##### tiles of constant size

Generate tiles of constant size with overlapping.

```bash
>  tile_generator const_size --help

Usage: tile_generator const_size [OPTIONS] INPUT_DIR_OR_FILE OUTPUT_DIR
                                 TILE_SIZE_IN_PIXELS MIN_OVERLAPPING_IN_PIXELS

Options:
  --extensions TEXT        String of file extensions to select (if input is a
                           directory), e.g. 'jpg,png,tif'
  --output_extension TEXT  Output tile file extension
  --n_workers INTEGER      Number of workers in the processing pool
                           [default=4]
  -q, --quiet              Disable verbose mode
  --help                   Show this message and exit.
```
For example,
```bash
> mkdir examples/tiles
> tile_generator const_size --extensions="jpg,png" examples/dog.jpg examples/tiles 256 20
```

