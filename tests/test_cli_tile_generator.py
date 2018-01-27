#
# Test tile generator
#
import tempfile
import shutil
import os
from glob import glob

from unittest import TestCase, main

import numpy as np

from osgeo.gdal import __version__ as gdal_version

from gimg.cli import get_files_from_folder, write_to_file, EXTENSIONS_GDAL_DRIVER_CODE_MAP
from gimg import GeoImage, GeoImageTilerConstSize
from gimg.cli.tile_generator import cli


from .create_datasets import create_dataset_with_target_is_folder, create_dataset_with_target_is_mask_file, \
    create_dataset_with_target_is_mask_file2, create_potsdam_like_dataset

from . import check_metadata


from click.testing import CliRunner


class TestTileGeneratorCli(TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        this_dir = os.path.dirname(__file__)
        self.filepath = os.path.join(this_dir, "..", "examples", "dog.jpg")
        self.runner = CliRunner()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def _test_const_size_on_dog_image(self, n_workers):
        result = self.runner.invoke(cli, ["const_size", self.filepath, self.tempdir,
                                          "256", "20",
                                          "--output_extension", "png",
                                          "--n_workers", "%i" % n_workers])
        self.assertEqual(result.exit_code, 0, repr(result) + "\n" + result.output)
        self.assertTrue(os.path.exists(os.path.join(self.tempdir, "dog_tiles")))
        tiles = glob(os.path.join(self.tempdir, "dog_tiles", "*.png"))
        self.assertEqual(len(tiles), 12, tiles)

        # Check data:
        gimage = GeoImage(self.filepath)
        tiles = GeoImageTilerConstSize(gimage, tile_size=(256, 256), min_overlapping=20)
        for true_data, x, y in tiles:
            tile_filename = os.path.join(self.tempdir, "dog_tiles", "tile_%i_%i.png" % (x, y))
            self.assertTrue(os.path.exists(tile_filename))
            gtile = GeoImage(tile_filename)
            data = gtile.get_data()
            self.assertEqual(data.shape, true_data.shape)
            self.assertLess(np.sum(np.abs(true_data - data)), 1e-10)

    def test_const_size_on_dog_image(self):
        self._test_const_size_on_dog_image(n_workers=4)

    def test_const_size_on_dog_image2(self):
        self._test_const_size_on_dog_image(n_workers=1)


if __name__ == "__main__":
    main()
