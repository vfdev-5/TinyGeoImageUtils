
#
# Test GeoImage
#
from unittest import TestCase, main
import tempfile
import shutil

# Numpy
import numpy as np

# GDAL
from osgeo.gdal import __version__ as gdal_version

# Project
from gimg import GeoImage
from gimg.common import get_dtype
from gimg.GeoImage import compute_geo_extent, compute_geo_transform

from .create_synthetic_images import create_synthetic_image_file, create_virt_image
from . import check_metadata


class TestGeoImage(TestCase):

    def setUp(self):
        self.gdal_version_major = int(gdal_version[0])
        # Create local temp directory
        self.local_temp_folder = tempfile.mkdtemp()

    def tearDown(self):
        # Delete temp directory
        shutil.rmtree(self.local_temp_folder)

    def test_with_synthetic_image(self):
        is_complex = False
        shape = (120, 100, 2)
        depth = 2
        filepath, data, geo_extent, metadata, geo_transform, epsg = create_synthetic_image_file(self.local_temp_folder,
                                                                                                shape, depth,
                                                                                                is_complex)
        gimage = GeoImage(filepath)
        if self.gdal_version_major > 1:
            self.assertTrue(check_metadata(metadata, gimage.metadata),
                            "{} vs {}".format(metadata, gimage.metadata))

        self.assertLess(np.sum(np.abs(geo_extent - gimage.geo_extent)), 1e-10)
        self.assertEqual(epsg, gimage.get_epsg())
        gimage_data = gimage.get_data()
        self.assertEqual(shape, gimage_data.shape)
        self.assertEqual(get_dtype(depth, is_complex), gimage_data.dtype)
        # verify data
        self.assertLess(np.sum(np.abs(data - gimage_data)), 1e-10)

    def test_with_synthetic_image_with_select_bands(self):
        is_complex = False
        shape = (120, 100, 5)
        depth = 2
        filepath, data, geo_extent, metadata, geo_transform, epsg = create_synthetic_image_file(self.local_temp_folder,
                                                                                                shape, depth,
                                                                                                is_complex)

        gimage = GeoImage(filepath)
        if self.gdal_version_major > 1:
            self.assertTrue(check_metadata(metadata, gimage.metadata),
                            "{} vs {}".format(metadata, gimage.metadata))
        self.assertLess(np.sum(np.abs(geo_extent - gimage.geo_extent)), 1e-10)
        self.assertEqual(epsg, gimage.get_epsg())
        select_bands = [0, 2, 4]
        gimage_data = gimage.get_data(select_bands=select_bands)
        self.assertEqual(shape[:2], gimage_data.shape[:2])
        self.assertEqual(len(select_bands), gimage_data.shape[2])
        self.assertEqual(get_dtype(depth, is_complex), gimage_data.dtype)
        # verify data
        self.assertLess(np.sum(np.abs(data[:, :, select_bands] - gimage_data)), 1e-10)

    def test_with_virtual_image(self):

        dataset, data = create_virt_image(100, 120, 2, np.uint16)
        gimage = GeoImage.from_dataset(dataset)

        gimage_data = gimage.get_data(nodata_value=0)

        # verify shape and dtype:
        self.assertEqual(data.shape, gimage_data.shape)
        self.assertEqual(data.dtype, gimage_data.dtype)

        # verify data
        self.assertLess(np.sum(np.abs(data - gimage_data)), 1e-10)

    def test_with_virtual_image2(self):

        dataset, data = create_virt_image(100, 120, 2, np.float32)
        gimage = GeoImage.from_dataset(dataset)

        gimage_data = gimage.get_data(nodata_value=-123)

        # verify shape and dtype:
        self.assertEqual(data.shape, gimage_data.shape)
        self.assertEqual(data.dtype, gimage_data.dtype)

        # verify data
        self.assertLess(np.sum(np.abs(data - gimage_data)), 1e-10)

    def test_from_dataset_with_select_bands(self):

        dataset, data = create_virt_image(100, 120, 5, np.float32)
        gimage = GeoImage.from_dataset(dataset)

        select_bands = [0, 2, 4]
        gimage_data = gimage.get_data(nodata_value=-123, select_bands=select_bands)

        # verify shape and dtype:
        self.assertEqual(data.shape[:2], gimage_data.shape[:2])
        self.assertEqual(len(select_bands), gimage_data.shape[2])
        self.assertEqual(data.dtype, gimage_data.dtype)

        # verify data
        self.assertLess(np.sum(np.abs(data[:, :, select_bands] - gimage_data)), 1e-10)

    def test_compute_geo_transform(self):
        geo_extent = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
        true_geo_transform = [0.0, 1.0 / 99.0, -1.0 / 99.0, 1.0, -1.0 / 99.0, -1.0 / 99.0]
        geo_transform = compute_geo_transform(geo_extent, (100, 100))
        self.assertLess(np.sum(np.abs(geo_transform - true_geo_transform)), 1e-10,
                        "{} vs {}".format(geo_transform, true_geo_transform))

    def test_compute_geo_extent(self):
        geo_transform = [0.0, 1.0 / 99.0, -1.0 / 99.0, 1.0, -1.0 / 99.0, -1.0 / 99.0]
        true_geo_extent = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
        geo_extent = compute_geo_extent(geo_transform, (100, 100))
        self.assertLess(np.sum(np.abs(geo_extent - true_geo_extent)), 1e-10,
                        "{} vs {}".format(geo_extent, true_geo_extent))


if __name__ == "__main__":
    main()
