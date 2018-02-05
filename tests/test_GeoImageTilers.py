# Test GeoImageTilers

import os
import shutil
import tempfile
from unittest import TestCase, main

import numpy as np

from gimg import GeoImage
from gimg import GeoImageTiler, GeoImageTilerConstSize
from gimg.GeoImage import from_ndarray
from gimg.GeoImageTilers import BaseGeoImageTiler
from .create_synthetic_images import create


class TestBaseGeoImageTiler(TestCase):

    def setUp(self):
        # Create local temp directory
        self.local_temp_folder = tempfile.mkdtemp()

        filepath = os.path.join(self.local_temp_folder, 'test_small_image.tif')
        shape = (120, 100, 2)
        depth = 2
        is_complex = False
        geo_transform = (12.345, 0.001, 0.0, 23.456, 0.0, -0.001)
        create(shape[1], shape[0], shape[2], filepath,
               depth=depth, is_complex=is_complex,
               geo_transform=geo_transform, epsg=4326)

        self.assertTrue(os.path.exists(filepath))
        self.geo_image = GeoImage(filepath)

    def tearDown(self):
        # Delete temp directory
        shutil.rmtree(self.local_temp_folder)

    def test_compute_geo_extent(self):
        h, w, c = self.geo_image.shape
        total_geo_extent = self.geo_image.geo_extent

        def _test(extent, scale):
            tiles = BaseGeoImageTiler(self.geo_image, tile_size=(32, 32), scale=scale)
            geo_extent = tiles.compute_geo_extent(extent)
            self.assertLess(np.sum(np.abs(total_geo_extent - geo_extent)), 1e-10)

        _test((0, 0, w, h), scale=1.0)
        # _test((0, 0, w // 2, h // 2), scale=2.0)


class TestGeoImageTiler(TestCase):

    def setUp(self):
        # Create local temp directory
        self.local_temp_folder = tempfile.mkdtemp()

        filepath = os.path.join(self.local_temp_folder, 'test_small_image.tif')
        metadata = {'key_1': 'value_1', 'key_2': "1 2 3", 'key_3': '3'}
        shape = (120, 100, 2)
        depth = 2
        is_complex = False
        geo_transform = (13.60746033, 0.001, 0.0, 50.25013288, 0.0, -0.001)
        create(shape[1], shape[0], shape[2], filepath,
               depth=depth, is_complex=is_complex,
               metadata=metadata, geo_transform=geo_transform, epsg=4326)

        self.assertTrue(os.path.exists(filepath))
        self.geo_image = GeoImage(filepath)

    def tearDown(self):
        # Delete temp directory
        shutil.rmtree(self.local_temp_folder)

    def test_tiling_no_overlapping_no_nodata(self):

        def _test(tile_size):
            tiled_image = np.zeros(self.geo_image.shape)
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size, overlapping=0)
            self.assertTrue(tiles.nx, int(np.ceil(tiled_image.shape[1] / tile_size[0])))
            self.assertTrue(tiles.ny, int(np.ceil(tiled_image.shape[0] / tile_size[1])))
            for tile, (x, y, w, h) in tiles:
                tiled_image[y:y + tile_size[1], x:x + tile_size[0], :] = tile

            err = float(np.sum(np.abs(tiled_image - self.geo_image.get_data())))
            self.assertLess(err, 1e-10, "Err : %f" % err)

        _test((32, 32))
        _test((40, 40))
        _test((44, 46))
        _test((55, 76))

    def test_tiling_no_overlapping_no_nodata_with_scale(self):

        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        for i in range(10):
            for j in range(10):
                test_image[10 * i:10 * i + 10, 10 * j:10 * j + 10, :] = np.random.randint(50, 256, size=(3,))

        def _test(tile_size, scale):
            gimage = from_ndarray(test_image)
            tiles = GeoImageTiler(gimage, tile_size=tile_size,
                                  overlapping=0, scale=scale, resample_alg=0)
            for tile, (x, y, w, h) in tiles:
                true_data = gimage.get_data([x, y, w, h],
                                            dst_width=tile_size[0],
                                            dst_height=tile_size[1],
                                            resample_alg=0)
                true_data = true_data[:tile.shape[0], :tile.shape[1]]

                err = np.sum(np.abs(true_data - tile))
                self.assertLess(err, 1e-10,
                                "Error : {} | conf: {}, {}".format(err, tile_size, scale))

        for tile_size in [(20, 20), (30, 30), (20, 30)]:
            for scale in [1.2, 1.5, 2.0, 0.7, 0.5, 0.2]:
                _test(tile_size=tile_size, scale=scale)

    def test_tiling_no_overlapping_with_nodata(self):

        def _test(tile_size):
            overlapping = 0
            h = int(np.ceil(self.geo_image.shape[0] * 1.0 / tile_size[1])) * tile_size[1]
            w = int(np.ceil(self.geo_image.shape[1] * 1.0 / tile_size[0])) * tile_size[0]
            nc = self.geo_image.shape[2]
            tiled_image = np.zeros((h, w, nc))
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size, overlapping=overlapping, nodata_value=0)
            for tile, (x, y, w, h) in tiles:
                tiled_image[y:y + tile_size[1], x:x + tile_size[0], :] = tile

            h, w, nc = self.geo_image.shape
            err = float(np.sum(np.abs(tiled_image[:h, :w, :] - self.geo_image.get_data())))
            self.assertLess(err, 1e-10, "Err : %f" % err)

        _test((32, 32))
        _test((40, 40))
        _test((44, 46))
        _test((55, 76))

    def test_tiling_with_overlapping_no_nodata(self):

        def _test(tile_size, overlapping):
            h, w, nc = self.geo_image.shape
            tiled_image = np.zeros((h, w, nc))
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size, overlapping=overlapping)
            for tile, (x, y, w, h) in tiles:
                if x == 0:
                    xend = tile_size[0] - overlapping
                else:
                    xend = min(x + tile_size[0], w)

                if y == 0:
                    yend = tile_size[1] - overlapping
                else:
                    yend = min(y + tile_size[1], h)

                tiled_image[y:yend, x:xend, :] = tile

            self.assertTrue(tiled_image.shape == self.geo_image.shape)
            err = float(np.sum(np.abs(tiled_image - self.geo_image.get_data())))
            self.assertLess(err, 1e-10, "Err : %f" % err)

        _test((32, 32), 8)
        _test((32, 32), 12)
        _test((32, 32), 13)
        _test((40, 40), 8)
        _test((40, 40), 11)
        _test((40, 40), 14)
        _test((44, 46), 7)
        _test((44, 46), 20)
        _test((44, 46), 11)
        _test((55, 76), 8)
        _test((55, 76), 24)
        _test((55, 76), 15)

    def test_tiling_with_overlapping_with_nodata(self):

        def _test(tile_size, overlapping):
            ny = int(np.ceil((self.geo_image.shape[0] + overlapping) * 1.0 / (tile_size[1] - overlapping)))
            nx = int(np.ceil((self.geo_image.shape[1] + overlapping) * 1.0 / (tile_size[0] - overlapping)))
            h = ny * tile_size[1]
            w = nx * tile_size[0]
            nc = self.geo_image.shape[2]
            tiled_image = np.zeros((h, w, nc))
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size, overlapping=overlapping, nodata_value=0)
            self.assertTrue(tiles.nx, nx)
            self.assertTrue(tiles.ny, ny)
            for tile, (x, y, w, h) in tiles:
                x += overlapping
                xend = x + tile_size[0]
                y += overlapping
                yend = y + tile_size[1]

                tiled_image[y:yend, x:xend, :] = tile

            h, w, nc = self.geo_image.shape
            x = overlapping
            xend = x + w
            y = overlapping
            yend = y + h
            err = float(np.sum(np.abs(tiled_image[y:yend, x:xend, :] - self.geo_image.get_data())))
            self.assertLess(err, 1e-10, "Err : %f" % err)

        _test((32, 32), 8)
        _test((32, 32), 12)
        _test((32, 32), 13)
        _test((40, 40), 8)
        _test((40, 40), 11)
        _test((40, 40), 14)
        _test((44, 46), 7)
        _test((45, 46), 22)
        _test((44, 46), 11)
        _test((55, 76), 8)
        _test((55, 76), 24)
        _test((55, 76), 15)

    def test_tiling_with_overlapping_with_nodata_with_scale(self):
        def _test(tile_size, overlapping, scale, nodata):

            h = int(self.geo_image.shape[0] * scale)
            w = int(self.geo_image.shape[1] * scale)
            ny = int(np.ceil((w + overlapping) * 1.0 / (tile_size[1] - overlapping)))
            nx = int(np.ceil((h + overlapping) * 1.0 / (tile_size[0] - overlapping)))
            h2 = ny * tile_size[1]
            w2 = nx * tile_size[0]
            nc = self.geo_image.shape[2]
            tiled_image = np.zeros((h2, w2, nc)) + nodata
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size,
                                  overlapping=overlapping,
                                  scale=scale,
                                  nodata_value=nodata)
            self.assertTrue(tiles.nx, nx)
            self.assertTrue(tiles.ny, ny)

            for tile, (x, y, w, h) in tiles:
                x = int(x * scale + overlapping)
                xend = x + min(tile_size[0], tile.shape[1])
                y = int(y * scale + overlapping)
                yend = y + min(tile_size[1], tile.shape[0])
                tiled_image[y:yend, x:xend, :] = tile

            x = overlapping
            xend = x + w
            y = overlapping
            yend = y + h
            true_data = self.geo_image.get_data(dst_width=w, dst_height=h)
            err = float(np.sum(np.abs(tiled_image[y:yend, x:xend] - true_data)))
            self.assertLess(err, 1e-10,
                            "Error : {} | conf: {}, {}, {}, {}".format(err, tile_size, overlapping, scale, nodata))

            # for tile_size in [(32, 32), (40, 40), (32, 40)]:
            #     for overlapping in [8, 12, 13, 15]:
            #         for scale in [0.5, 0.2, 1.2, 1.5]:
            #             for nodata in [0, -1, 255, -1.0]:
            #                 _test(tile_size, overlapping, scale, nodata)


class TestGeoImageTilerConstSize(TestCase):

    def setUp(self):
        # Create local temp directory
        self.local_temp_folder = tempfile.mkdtemp()

        filepath = os.path.join(self.local_temp_folder, 'test_small_image.tif')
        metadata = {'key_1': 'value_1', 'key_2': "1 2 3", 'key_3': '3'}
        shape = (120, 100, 2)
        depth = 2
        is_complex = False
        geo_transform = (13.60746033, 0.001, 0.0, 50.25013288, 0.0, -0.001)
        create(shape[1], shape[0], shape[2], filepath,
               depth=depth, is_complex=is_complex,
               metadata=metadata, geo_transform=geo_transform, epsg=4326)

        self.assertTrue(os.path.exists(filepath))
        self.geo_image = GeoImage(filepath)

    def tearDown(self):
        # Delete temp directory
        shutil.rmtree(self.local_temp_folder)

    def test_tiling_no_min_overlapping(self):

        def _test(tile_size):
            tiled_image = np.zeros(self.geo_image.shape)
            tiles = GeoImageTilerConstSize(self.geo_image, tile_size=tile_size, min_overlapping=0)
            self.assertTrue(tiles.nx, int(np.ceil(tiled_image.shape[1] / tile_size[0])))
            self.assertTrue(tiles.ny, int(np.ceil(tiled_image.shape[0] / tile_size[1])))
            for tile, (x, y, w, h) in tiles:
                self.assertTrue(tile.shape[:2] == (tile_size[1], tile_size[0]))
                tiled_image[y:y + tile_size[1], x:x + tile_size[0], :] = tile

            err = float(np.sum(np.abs(tiled_image - self.geo_image.get_data())))
            self.assertLess(err, 1e-10, "Err : %f" % err)

        _test((32, 32))
        _test((40, 40))
        _test((44, 46))
        _test((55, 76))

    def test_tiling_with_min_overlapping(self):

        def _test(tile_size, overlapping):
            h, w, nc = self.geo_image.shape
            tiled_image = np.zeros((h, w, nc))
            tiles = GeoImageTilerConstSize(self.geo_image, tile_size=tile_size, min_overlapping=overlapping)
            for tile, (x, y, w, h) in tiles:
                self.assertTrue(tile.shape[:2] == (tile_size[1], tile_size[0]))
                tiled_image[y:y + tile_size[1], x:x + tile_size[0], :] = tile

            self.assertTrue(tiled_image.shape == self.geo_image.shape)
            err = float(np.sum(np.abs(tiled_image - self.geo_image.get_data())))
            self.assertLess(np.abs(err), 1e-10, "Err : %f" % err)

        _test((32, 32), 8)
        _test((32, 32), 12)
        _test((32, 32), 13)
        _test((40, 40), 8)
        _test((40, 40), 11)
        _test((40, 40), 14)
        _test((44, 46), 7)
        _test((44, 46), 20)
        _test((44, 46), 11)
        _test((55, 76), 8)
        _test((55, 76), 24)
        _test((55, 76), 15)


if __name__ == "__main__":
    main()
