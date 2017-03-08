# Tiny Geographical Image Utils

Reading and manipulation of geographical images. Based on GDAL

## Basic tools:

* `GeoImage` for reading geographical images 

* `GeoImageTiler` for tiled reading geographical images


## Basic usage:

```
import numpy
import matplotlib.pyplot as plt
from gimg_utils.GeoImage import GeoImage

gimg = GeoImage("path/to/image/file")
np_img = gimg.get_data([0, 0, 500, 500])

print np_img.shape, np_img.type
plt.imshow(np_img[:,:,0])
plt.show()
```

and 

```
import numpy
import matplotlib.pyplot as plt
from gimg_utils.GeoImage import GeoImage
from gimg_utils.GeoImage import GeoImageTilerConstSize

gimg = GeoImage("path/to/image/file")
tiles = GeoImageTilerConstSize(gimg, tile_size=(512, 512), min_overlapping=128)

for tile, x, y in tiles:
    print tile.shape, tile.type, x, y
```

See [example.ipynb](examples/examples.ipynb) for some basic examples of usage