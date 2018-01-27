#
# Example of geo image files creation from buffer or whatever
#

import numpy as np

from gimg.cli import write_to_file


if __name__ == "__main__":

    metadata = {'Description': 'A random rectangle on the Earth'}
    geo_info = {
        'epsg': 4326,
        'geo_extent': [[1.1, 43.0], [1.12, 43.0], [1.12, 42.99], [1.1, 42.99]]
    }
    data1 = np.zeros((256, 256, 3), dtype=np.uint8)
    data1[50:200, 50:200, :] = (128, 180, 220)
    data2 = np.zeros((256, 256, 3), dtype=np.uint16)
    data2[70:220, 70:220, :] = (220, 180, 120)

    write_to_file(data1, "test.png", geo_info, metadata)
    write_to_file(data2, "test.tif", geo_info, metadata, options=["COMPRESS=LZW"])
    write_to_file(data1, "test.jpg", geo_info, metadata, options=["QUALITY=100"])
