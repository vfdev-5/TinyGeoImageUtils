# -*- coding:utf-8 -*-
from __future__ import absolute_import

# Python
import logging

# Numpy
import numpy as np

# Project
from .GeoImage import GeoImage

logger = logging.getLogger('gimg')


class BaseGeoImageTiler(object):
    """
    Base class to tile a GeoImage
    See the implementations
        - GeoImageTiler
        - GeoImageTilerConstSize

    Essential parameters:
    :param geo_image: instance of GeoImage
    :param tile_size: (list or tuple) output tile size in pixels
    :param scale: (float) Scale of the tile extent. Default, scale is 1. Scale corresponds to a factor
        between requested ROI from the image and returned tile.
        For example, when scale=0.5 returned tiles represent two times larger zone in pixels than the same zone in the
        image, e.g source image rectangle `[x, y, tile_size_x / scale, tile_size_y / scale]` is sampled to a tile of
        size `tile_size=(tile_size_x, tile_size_y)`
    :param tile_origin: (list or tuple) point from where to start the tiling. Values should be given in pixels of
        the input image. If scale is applied, tile_origin is not impacted.
    :param resample_alg: type of resample algorithm to use if a scale is applied.
        Possible values:
            - gdalconst.GRIORA_NearestNeighbour (default)
            - gdalconst.GRIORA_Bilinear
            - gdalconst.GRIORA_Cubic
            - gdalconst.GRIORA_CubicSpline
            - gdalconst.GRIORA_Lanczos
            - gdalconst.GRIORA_Gauss
            - gdalconst.GRIORA_Average
    """

    def __init__(self, geo_image, tile_size=(1024, 1024), scale=1.0, tile_origin=(0, 0), resample_alg=0):
        assert isinstance(geo_image, GeoImage), "Argument geo_image should be instance of GeoImage"
        assert isinstance(tile_size, (tuple, list)) and len(tile_size) == 2, \
            "Argument tile_size should be a tuple (sx, sy)"
        assert scale > 0.0, "Argument scale should be positive"
        assert isinstance(tile_origin, (tuple, list)) and len(tile_origin) == 2, \
            "Argument tile_origin should be a tuple (px, py)"
        for o, s in zip(tile_origin, tile_size):
            assert -o < s, "Negative tile origin should not be larger than tile size"

        self._geo_image = geo_image
        self.tile_size = tile_size
        self.tile_origin = tile_origin
        self.scale = float(scale)
        self._index = 0
        self._maxIndex = 0
        self.nodata_value = 0
        self.resample_alg = resample_alg
        self._tile_extent = [int(d / self.scale) for d in self.tile_size]

    def __iter__(self):
        return self

    def get_lin_index(self):
        return self._index - 1

    def compute_geo_extent(self, extent):
        """
        Compute geo extent of a rectangle in the current configuration.
        This method is useful to compute tile geo extent
        ```
        # tiles defined with any scale
        for tile, tile_extent in tiles:
            ge = tiles.compute_geo_extent(tile_extent)
        ```
        :param extent: rectangle extent [x, y, w, h] in the original image, in pixels
        :return: ndarray of rectangle geo extent in lat/lon
            [[top-left_x, top-left_y],
            [top-right_x, top-right_y],
            [bottom-left_x, bottom-left_y],
            [bottom-right_x, bottom-right_y]]
        """
        assert isinstance(extent, (list, tuple)) and len(extent) == 4, \
            "Argument extent should be a list or tuple of size 4, [x, y, w, h]"
        x, y, w, h = extent
        points = np.array([[x, y],
                           [x + w - 1, y],
                           [x + w - 1, y + h - 1],
                           [x, y + h - 1]])
        return self._geo_image.transform(points)

    # def _get_size_extent(self, size):
    #     """
    #     Compute pixel extent of a rectangle on the original image
    #     If scale is 1, output is identical to the input
    #     If scale is 0.5, output is 2 times larger than the input
    #     :param size: width, height in pixels
    #     :return: [size_extent_x, size_extent_y] in pixels
    #     """
    #     return [int(size[0] * self.scale), size[1] * self.scale]

    def _get_current_tile_extent(self):
        raise Exception("Not implemented")

    def next(self):
        """
            Method to get current tile
            :return: tile data (ndarray), tile extent (list) in the original image, in pixels
        """
        if self._index < 0 or self._index >= self._maxIndex:
            raise StopIteration()

        # Define ROI to extract
        tile_extent, tile_size, x_tile_index, y_tile_index = self._get_current_tile_extent()
        # tile_extent = [xoffset, yoffset, tile_extent_x, tile_extent_y] in original image (pixels)

        logger.debug("{}/{} = ({},{}) | extent={}, size={}"
                     .format(self._index, self._maxIndex, x_tile_index, y_tile_index, tile_extent, tile_size))

        # Extract data
        logger.debug("dst_width={}, dst_height={}".format(tile_size[0], tile_size[1]))

        data = self._geo_image.get_data(tile_extent,
                                        dst_width=tile_size[0],
                                        dst_height=tile_size[1],
                                        nodata_value=self.nodata_value,
                                        resample_alg=self.resample_alg)

        assert data is not None, "Ops, there is an internal problem to get tile data." + \
                                 "\ntile_extent={}".format(tile_extent) + \
                                 "\ntile_size={}".format(tile_size)
        logger.debug("data.shape={}".format(data.shape))
        # ++
        self._index += 1

        return data, tile_extent

    __next__ = next


class GeoImageTiler(BaseGeoImageTiler):
    """Helper class to iterate over GeoImage

    A tiles grid is created starting from `tile_origin` and covers the input image with tiles.
    Tiles of maximum size `(tile_size[0], tile_size[1])` are extracted with a four-sided `overlapping`

    There are two options to produce tile:

    a) `nodata_value=0`: All tiles have the same size `tile_size` and nodata values are filled with `nodata_value`.
        For example, we choose tile origin=(-overlapping, -overlapping), the first top-left tile starts outside the
        image:
        <--- tile_size[0] -->
      tile origin     overlapping
        <--->           <--->
         ____________________
        |NDV|NDV NDV NDV|NDV|
        |---*************---|
        |NDV*   *   *   *   |
        |NDV*   *   *   *   |
        |NDV*   *   *   *   |
        |---*************---|  ^
        |NDV|___________|___|  |overlapping
                               v
        and the last bottom-right tile can look like this :

        <--- tile_size[0] -->
      overlapping
        <--->
         ____________________  ^
        |   |       |NDV NDV|  | overlapping
        |---*********-------|  v
        |   *   *   *NDV NDV|
        |   *   *   *NDV NDV|
        |   *   *   *NDV NDV|
        |---*********-------|
        |NDV NDV NDV NDV NDV|

        In the case (a) the tile offset on the original image is computed as
        ```
        x_tile_offset = i*(tile_size[0] - overlapping) + x_tile_origin
        y_tile_offset = j*(tile_size[1] - overlapping) + y_tile_origin
        ```

        b) `nodata_value=None`: All tiles have the same size `tile_size`, except boundary tiles.
        For example, the first top-left tile looks like

        tile_size[0] - tile_origin[0]
        <--------------->
                   overlapping
                    <--->
        *************---|
        *   *   *   *   |
        *   *   *   *   |
        *   *   *   *   |
        *************---|
        |___________|___|

        Somewhere in the middle of the image, the tile has the size `tile_size` :

        <--- tile_size[0] -->
      overlapping     overlapping
        <--->           <--->
         ____________________
        |   |           |   |
        |---*************---|
        |   *   *   *   *   |
        |   *   *   *   *   |
        |   *   *   *   *   |
        |---*************---|
        |___|___________|___|

        and the last bottom-right tile is reduced to stop at the last pixel:

        <--- tile_size[0] -->
      overlapping
        <--->
         ____________
        |   |       |
        |---********|
        |   *   *   |
        |   *   *   |
        |   *   *   |
        |---********|


    In the case (b) the tile offset is computed as
    ```
    x_tile_offset = max(0, i*(tile_size[0] - overlapping) + x_tile_origin)
    y_tile_offset = max(0, j*(tile_size[1] - overlapping) + y_tile_origin)
    ```

    Usage :
    ```
        gimage = GeoImage('path/to/file')
        tiles = GeoImageTiler(gimage, tile_size=(1024, 1024), overlapping=256)

        for tile, xoffset, yoffset in tiles:
            assert isinstance(tile, np.ndarray), "..."
    ```
    """

    def __init__(self, geo_image, tile_size=(1024, 1024), overlapping=256, tile_origin=(0, 0),
                 nodata_value=None, scale=1.0, resample_alg=0):
        """
        Initialize tiler
        :param geo_image: (GeoImage) input image
        :param tile_size: (list or tuple) tile output size, e.g. (1024, 1024) in pixels
        :param overlapping: (int) four-sided overlapping between tiles in pixels. If scale is applied, overlapping is
            not impacted.
        :param tile_origin:
        :param nodata_value:
        :param scale:
        :param resample_alg:
        """
        super(GeoImageTiler, self).__init__(geo_image, tile_size, scale, tile_origin, resample_alg)
        assert 0 <= overlapping < min(tile_size[0], tile_size[1]), \
            "Argument overlapping should be less than the minimum of tile_size"

        self.overlapping = overlapping
        self.nx = GeoImageTiler._compute_number_of_tiles(self._tile_extent[0], self.tile_origin[0],
                                                         self._geo_image.shape[1], self.overlapping)
        self.ny = GeoImageTiler._compute_number_of_tiles(self._tile_extent[1], self.tile_origin[1],
                                                         self._geo_image.shape[0], self.overlapping)
        self._maxIndex = self.nx * self.ny
        self.nodata_value = nodata_value

    @staticmethod
    def _compute_number_of_tiles(tile_size, tile_origin, image_size, overlapping):
        """
            Method to compute number of overlapping tiles for a given image size
            n = ceil((imageSize - tile_origin)/(tileSize - overlapping ))
            imageSize :  [01234567891] (isFourSided=true), tileSize=6, overlapping=2, tile_origin=-2
            tile   0  :[xx0123]
                   1  :    [234567]
                   2  :        [67891x]
                   3  :            [1xxxxx]
              n = ceil ( (11+2) / (6 - 2) ) = 4

            imageSize :  [012345678901234] (isFourSided=true), tileSize=7, overlapping=2, tile_origin=-2
            tile   0  :[xx01234]
                   1  :     [3456789]
                   2  :          [8901234]
                   3  :               [34xxxxx]
              n = ceil ( (15 + 2) / (7 - 2) ) = ceil(3.4) = 4

            imageSize :  [012345678901234] (isFourSided=true), tileSize=7, overlapping=2, tile_origin=0
            tile   0  :  [0123456]
                   1  :       [5678901]
                   2  :            [01234xx]
              n = ceil ( (15) / (7 - 2) ) = 3
        """
        return ceil_int((image_size - tile_origin) * 1.0 / (tile_size - overlapping))

    @staticmethod
    def _compute_tile_params(index, tile_extent, tile_size, tile_origin, overlapping, scale, nodata_value, image_size):
        """
        Internal method to compute tile parameters: tile offset, tile_extent, output_size
        :param index:
        :param tile_extent:
        :param tile_size:
        :param overlapping:
        :param scale:
        :param nodata_value:
        :param image_size:
        :return:
        """
        tile_offset = index * (tile_extent - overlapping) - overlapping
        output_size = tile_size

        if nodata_value is None:
            tile_extent = min(tile_extent, image_size - tile_offset)
            if index == 0:
                tile_offset = 0
                tile_extent -= overlapping
            output_size = min(output_size, int(np.ceil(tile_extent * scale)))
        return tile_offset, tile_extent, output_size

    def _get_current_tile_extent(self):
        """
        isFourSided = true:
        tileSize = 6, overlapping = 1

        offset:  0    4   9    14   19
        Image:  [----------------------]
        Tiles: [x....O]                ]
                [   [O....O]           ]
                [        [O....O]      ]
                [             [O....O] ]
                [                  [O..xxx]

        offset(i) = {i == 0: 0,
                    {i > 0: i * (tileSize - overlapping) - overlapping

        size(i) = {i == 0: tileSize - overlapping,
                  {i > 0: offset(i) + tileSize < imageWidth ? tileSize: imageWidth - offset(i)

        bufferOffset(i) = {i == 0: dataPtr + overlapping
                          {i > 0: dataPtr + 0

        :return: tile_extent (in the original image), tile_size, x_tile_index, y_tile_index
        """
        x_tile_index = self._index % self.nx
        y_tile_index = int(self._index * 1.0 / self.nx)

        def _compute(index, tile_extent, tile_size, overlapping, scale, nodata_value, image_size):
            tile_offset = index * (tile_extent - overlapping) - overlapping
            output_size = tile_size

            if nodata_value is None:
                tile_extent = min(tile_extent, image_size - tile_offset)
                if index == 0:
                    tile_offset = 0
                    tile_extent -= overlapping
                output_size = min(output_size, int(tile_extent * scale + 1))

            return tile_offset, tile_extent, output_size

        x_tile_offset, x_tile_extent, x_tile_size = _compute(x_tile_index, self._tile_extent[0], self.tile_size[0],
                                                             self._overlapping_extent, self.scale, self.nodata_value,
                                                             self._geo_image.shape[1])
        y_tile_offset, y_tile_extent, y_tile_size = _compute(y_tile_index, self._tile_extent[1], self.tile_size[1],
                                                             self._overlapping_extent, self.scale, self.nodata_value,
                                                             self._geo_image.shape[0])

        return [x_tile_offset, y_tile_offset, x_tile_extent, y_tile_extent], \
               [x_tile_size, y_tile_size], \
               x_tile_index, y_tile_index

        # # Compute offset, tile extent on the original image and output tile size
        # def _compute(index, tile_size, overlapping, scale, nodata_value, image_size):
        #     tile_offset = index * (tile_size - overlapping) - overlapping
        #     nonscaled_tile_offset = tile_offset
        #     tile_offset = int(tile_offset / scale)
        #
        #     tile_extent = tile_size
        #     output_size = tile_size
        #
        #     if nodata_value is None:
        #         output_size = min(output_size, image_size - tile_offset)
        #         tile_extent = min(tile_extent, image_size - nonscaled_tile_offset)
        #         if index == 0:
        #             tile_offset = 0
        #             output_size -= overlapping
        #             tile_extent -= overlapping
        #
        #     tile_extent = int(tile_extent / scale)
        #     return tile_offset, tile_extent, output_size
        #
        # x_tile_offset, x_tile_extent, x_tile_size = _compute(x_tile_index, self.tile_size[0], self.overlapping,
        #                                                      self.scale, self.nodata_value, self._geo_image.shape[1])
        # y_tile_offset, y_tile_extent, y_tile_size = _compute(y_tile_index, self.tile_size[1], self.overlapping,
        #                                                      self.scale, self.nodata_value, self._geo_image.shape[0])

        # if self.nodata_value is None:
        #     image_width = self._geo_image.shape[1] * self.scale
        #     image_height = self._geo_image.shape[0] * self.scale
        #     x_tile_extent = min(image_width - x_tile_offset, x_tile_extent)
        #     y_tile_extent = min(image_height - y_tile_offset, y_tile_extent)

        # return [x_tile_offset, y_tile_offset, x_tile_extent, y_tile_extent], \
        #        [x_tile_size, y_tile_size], \
        #        x_tile_index, y_tile_index


class GeoImageTilerConstSize(BaseGeoImageTiler):
    """
        Helper class to iterate over GeoImage
        Tiles of size (tile_size[0], tile_size[1]) are extracted with a four-sided overlapping
        starting from the top-left corner and all tiles have the same size without going outbound of the image.
        This is done computing the horizontal and vertical overlappings to minimize the number of tiles needed to
        cover the image. User can also specify a minimum overlapping `min_overlapping`.

        For example, tiling can look like this:
          tile 0      tile 2      tile 4
        |<------>|  |<------>|  |<------>|
                tile 1      tile 3      tile 5
              |<------>|  |<------>|  |<------>|
        |<------------------------------------>|
                        IMAGE

        Usage :

            gimage = GeoImage('path/to/file')
            tiles = GeoImageTilerConstSize(gimage, tile_size=(1024, 1024), min_overlapping=256)

            for tile, xoffset, yoffset in tiles:
                assert instanceof(tile, np.ndarray), "..."

    """

    def __init__(self, geo_image, tile_size=(1024, 1024), min_overlapping=256, scale=1.0, resample_alg=0):

        super(GeoImageTilerConstSize, self).__init__(geo_image, tile_size, scale, resample_alg)
        assert 0 <= min_overlapping < min(tile_size[0], tile_size[1]), \
            "minimal overlapping should be between 0 and min tile_size"

        self.min_overlapping = min_overlapping
        h = self._geo_image.shape[0] * self.scale
        w = self._geo_image.shape[1] * self.scale

        self.nx = GeoImageTilerConstSize._compute_number_of_tiles(self.tile_size[0], w, min_overlapping)
        self.ny = GeoImageTilerConstSize._compute_number_of_tiles(self.tile_size[1], h, min_overlapping)
        self.float_overlapping_x = GeoImageTilerConstSize._compute_float_overlapping(self.tile_size[0], w,
                                                                                     self.nx)
        self.float_overlapping_y = GeoImageTilerConstSize._compute_float_overlapping(self.tile_size[1], h,
                                                                                     self.ny)
        self._maxIndex = self.nx * self.ny

    @staticmethod
    def _compute_number_of_tiles(tile_size, image_size, min_overlapping):
        """
            Method to compute number of overlapping tiles for a given image size
            n = ceil(image_size / (tile_size - min_overlapping))
        """
        return ceil_int(image_size * 1.0 / (tile_size - min_overlapping))

    @staticmethod
    def _compute_float_overlapping(tile_size, image_size, n):
        """
            Method to float overlapping

            delta = tile_size * n - image_size
            overlapping = delta / (n - 1)
        """
        return (tile_size * n - image_size) * 1.0 / (n - 1.0)

    def _get_current_tile_extent(self):
        """

        offset(i) = round(i * (tileSize - float_overlapping_x))
          size(i) = tileSize

        :return: current tile extent as
            [x_tile_offset, y_tile_offset, x_tile_extent, y_tile_extent],
            [x_tile_size, y_tile_size],
            x_tile_index, y_tile_index
        """
        x_tile_index = self._index % self.nx
        y_tile_index = int(self._index * 1.0 / self.nx)

        # Compute offset, tile extent on the original image and output tile size
        def _compute(index, tile_size, overlapping, scale):
            tile_offset = int(np.round(index * (tile_size - overlapping)))
            output_size = tile_size
            tile_offset = int(tile_offset / scale)
            tile_extent = int(output_size / scale)
            return tile_offset, tile_extent, output_size

        x_tile_offset, x_tile_extent, x_tile_size = _compute(x_tile_index, self.tile_size[0],
                                                             self.float_overlapping_x, self.scale)
        y_tile_offset, y_tile_extent, y_tile_size = _compute(y_tile_index, self.tile_size[1],
                                                             self.float_overlapping_y, self.scale)
        return [x_tile_offset, y_tile_offset, x_tile_extent, y_tile_extent], \
               [x_tile_size, y_tile_size], \
               x_tile_index, y_tile_index


def ceil_int(x):
    return int(np.ceil(x))
