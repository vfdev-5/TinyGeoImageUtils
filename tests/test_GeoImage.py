
#
# Test GeoImage
#
from unittest import TestCase, TestLoader, TextTestRunner
import tempfile
import shutil

# Numpy
import numpy as np

# Project
from gimg import GeoImage
from gimg.common import get_dtype

from .create_synthetic_images import create_synthetic_image_file, create_virt_image


class TestGeoImage(TestCase):

    def setUp(self):
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
        # Must add this metadata : 'IMAGE_STRUCTURE__INTERLEAVE': 'PIXEL', 'AREA_OR_POINT': 'Area'
        metadata['IMAGE_STRUCTURE__INTERLEAVE'] = 'PIXEL'
        metadata['AREA_OR_POINT'] = 'Area'
        self.assertEqual(metadata, gimage.metadata)
        self.assertTrue((geo_extent == gimage.geo_extent).all(),
                        "Wrong geo extent : {} != {}".format(geo_extent, gimage.geo_extent))
        self.assertEqual(epsg, gimage.get_epsg())
        gimage_data = gimage.get_data()
        self.assertEqual(shape, gimage_data.shape)
        self.assertEqual(get_dtype(depth, is_complex), gimage_data.dtype)
        # verify data
        self.assertEqual(float(np.sum(data - gimage_data)), 0.0)

    def test_with_synthetic_image_with_select_bands(self):
        is_complex = False
        shape = (120, 100, 5)
        depth = 2
        filepath, data, geo_extent, metadata, geo_transform, epsg = create_synthetic_image_file(self.local_temp_folder,
                                                                                                shape, depth,
                                                                                                is_complex)

        gimage = GeoImage(filepath)
        # Must add this metadata : 'IMAGE_STRUCTURE__INTERLEAVE': 'PIXEL', 'AREA_OR_POINT': 'Area'
        metadata['IMAGE_STRUCTURE__INTERLEAVE'] = 'PIXEL'
        metadata['AREA_OR_POINT'] = 'Area'
        self.assertEqual(metadata, gimage.metadata)
        self.assertTrue((geo_extent == gimage.geo_extent).all(),
                        "Wrong geo extent : {} != {}".format(geo_extent, gimage.geo_extent))
        self.assertEqual(epsg, gimage.get_epsg())
        select_bands = [0, 2, 4]
        gimage_data = gimage.get_data(select_bands=select_bands)
        self.assertEqual(shape[:2], gimage_data.shape[:2])
        self.assertEqual(len(select_bands), gimage_data.shape[2])
        self.assertEqual(get_dtype(depth, is_complex), gimage_data.dtype)
        # verify data
        self.assertEqual(float(np.sum(data[:, :, select_bands] - gimage_data)), 0.0)

    def test_with_virtual_image(self):

        dataset, data = create_virt_image(100, 120, 2, np.uint16)
        gimage = GeoImage.from_dataset(dataset)

        gimage_data = gimage.get_data(nodata_value=0)

        # verify shape and dtype:
        self.assertEqual(data.shape, gimage_data.shape)
        self.assertEqual(data.dtype, gimage_data.dtype)

        # verify data
        self.assertEqual(float(np.sum(data - gimage_data)), 0.0)

    def test_with_virtual_image2(self):

        dataset, data = create_virt_image(100, 120, 2, np.float32)
        gimage = GeoImage.from_dataset(dataset)

        gimage_data = gimage.get_data(nodata_value=-123)

        # verify shape and dtype:
        self.assertEqual(data.shape, gimage_data.shape)
        self.assertEqual(data.dtype, gimage_data.dtype)

        # verify data
        self.assertEqual(float(np.sum(data - gimage_data)), 0.0)

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
        self.assertEqual(float(np.sum(data[:, :, select_bands] - gimage_data)), 0.0)


if __name__ == "__main__":

    suite = TestLoader().loadTestsFromTestCase(TestGeoImage)
    TextTestRunner().run(suite)
