#
# Test common module
#
import tempfile
import shutil
import os

from unittest import TestCase, main

import numpy as np

from osgeo.gdal import __version__ as gdal_version

from gimg.cli import get_files_from_folder, write_to_file, EXTENSIONS_GDAL_DRIVER_CODE_MAP
from gimg import GeoImage


from .create_datasets import create_dataset_with_target_is_folder, create_dataset_with_target_is_mask_file, \
    create_dataset_with_target_is_mask_file2, create_potsdam_like_dataset

from . import check_metadata


class TestCliModule(TestCase):

    def setUp(self):
        self.gdal_version_major = int(gdal_version[0])
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_extension_gdal_driver_code_map(self):
        common_exts = ['tif', 'jpg', 'png', 'bmp', 'gif']
        for ext in common_exts:
            self.assertIn(ext, EXTENSIONS_GDAL_DRIVER_CODE_MAP)

    def test_get_files_from_folder(self):

        def _test(fn, extensions, **kwargs):
            p = os.path.join(self.tempdir, 'data')
            os.mkdir(p)
            true_dataset = fn(p, **kwargs)
            files = get_files_from_folder(self.tempdir, extensions=extensions)
            for img_filepath, _ in true_dataset:
                self.assertIn(img_filepath, files)
            shutil.rmtree(p)

        _test(create_dataset_with_target_is_folder, ("ext1", ), n=100)
        _test(create_dataset_with_target_is_folder, None, n=100)
        _test(create_dataset_with_target_is_mask_file, ("ext1", "ext2"), n=100)
        _test(create_dataset_with_target_is_mask_file2, ("ext1", "ext2"), n=100)
        _test(create_dataset_with_target_is_mask_file2, None, n=100)
        _test(create_potsdam_like_dataset, ("tif", ))

    def _generate_data_to_write(self, filename, dtype, shape):
        filepath = os.path.join(self.tempdir, filename)
        metadata = {'key_1': 'value_1', 'key_2': "1 2 3", 'key_3': '3'}
        geo_info = {
            'epsg': 4326,
            'geo_extent': [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
        }
        data = np.random.randint(0, 128, size=shape, dtype=dtype)
        return filepath, data, geo_info, metadata

    def _test_write_to_file(self, ext, dtype, shape, options=None):
        filepath, data, geo_info, metadata = self._generate_data_to_write(ext, dtype, shape)
        geo_extent = geo_info['geo_extent']

        write_to_file(data, filepath, geo_info, metadata, options=options)
        self.assertTrue(os.path.exists(filepath))

        gimage = GeoImage(filepath)

        if self.gdal_version_major > 1:
            self.assertTrue(check_metadata(metadata, gimage.metadata),
                            "{} vs {}".format(metadata, gimage.metadata))

        self.assertLess(np.sum(np.abs(geo_extent - gimage.geo_extent)), 1e-10,
                        "{} vs {}".format(geo_extent, gimage.geo_extent))

        gimage_data = gimage.get_data()
        self.assertEqual(shape, gimage_data.shape)
        self.assertEqual(dtype, gimage_data.dtype)
        # verify data
        self.assertLess(np.sum(np.abs(data - gimage_data)), 1e-10)

    def _test_write_to_file_no_geo(self, ext, dtype, shape, options=None):
        filepath, data, _, _ = self._generate_data_to_write(ext, dtype, shape)

        write_to_file(data, filepath, None, None, options=options)
        self.assertTrue(os.path.exists(filepath))
        gimage = GeoImage(filepath)
        gimage_data = gimage.get_data()
        self.assertEqual(shape, gimage_data.shape)
        self.assertEqual(dtype, gimage_data.dtype)
        # verify data
        self.assertLess(np.sum(np.abs(data - gimage_data)), 1e-10)

    def test_write_to_file(self):
        self._test_write_to_file('f1.tif', np.uint8, (50, 50, 3))
        self._test_write_to_file('f1c.tif', np.uint8, (50, 50, 3), options=['COMPRESS=LZW'])
        self._test_write_to_file('f2.tif', np.uint16, (50, 50, 4))
        self._test_write_to_file('f1.png', np.uint8, (50, 50, 3))
        # Can not test JPG on data exactness due to compression
        # self._test_write_to_file('f1.jpg', np.uint8, (50, 50, 3), options=['QUALITY=100'])

        self._test_write_to_file_no_geo('fng1.tif', np.uint8, (50, 50, 3))
        self._test_write_to_file_no_geo('fng1c.tif', np.uint8, (50, 50, 3), options=['COMPRESS=LZW'])
        self._test_write_to_file_no_geo('fng2.tif', np.uint16, (50, 50, 4))
        self._test_write_to_file_no_geo('fng1.png', np.uint8, (50, 50, 3))


class TestTileGeneratorCli(TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.filepath = os.path.join("..", "examples", "dog.jpg")

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_(self):
        pass


if __name__ == "__main__":
    main()
