#
# Test common module
#
import tempfile
import shutil
import os

from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np

from gimg.cli import get_files_from_folder, write_to_file, EXTENSIONS_GDAL_DRIVER_CODE_MAP
from gimg import GeoImage


from create_datasets import create_dataset_with_target_is_folder, create_dataset_with_target_is_mask_file, \
    create_dataset_with_target_is_mask_file2, create_potsdam_like_dataset

from create_synthetic_images import compute_geo_extent


class TestCliModule(TestCase):

    def setUp(self):
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
            'geo_transform': (13.60746033, 0.001, 0.0, 50.25013288, 0.0, -0.001)
        }
        data = np.random.randint(0, 128, size=shape, dtype=dtype)
        geo_extent = compute_geo_extent(geo_info['geo_transform'], shape)
        return filepath, data, geo_info, metadata, geo_extent

    def _test_write_to_file(self, ext, dtype, shape):
        filepath, data, geo_info, metadata, geo_extent = self._generate_data_to_write(ext, dtype, shape)
        write_to_file(data, filepath, geo_info, metadata)
        self.assertTrue(os.path.exists(filepath))

        gimage = GeoImage(filepath)
        # Must add this metadata : 'IMAGE_STRUCTURE__INTERLEAVE': 'PIXEL', 'AREA_OR_POINT': 'Area'
        metadata['IMAGE_STRUCTURE__INTERLEAVE'] = 'PIXEL'
        metadata['AREA_OR_POINT'] = 'Area'
        self.assertEqual(metadata, gimage.metadata)
        self.assertTrue((geo_extent == gimage.geo_extent).all(),
                        "Wrong geo extent : {} != {}".format(geo_extent, gimage.geo_extent))

        gimage_data = gimage.get_data()
        self.assertEqual(shape, gimage_data.shape)
        self.assertEqual(dtype, gimage_data.dtype)
        # verify data
        self.assertEqual(float(np.sum(data - gimage_data)), 0.0)

    def test_write_to_file(self):
        self._test_write_to_file('f1.tif', np.uint8, (50, 50, 3))
        self._test_write_to_file('f2.tif', np.uint16, (50, 50, 4))


if __name__ == "__main__":

    suite = TestLoader().loadTestsFromTestCase(TestCliModule)
    TextTestRunner().run(suite)

