#
# Test common module
#
from __future__ import absolute_import

from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np
import gdal

from gimg.common import get_basename, get_dtype, get_gdal_dtype, gdal_to_numpy_datatype


class TestCommon(TestCase):

    def test_get_basename(self):
        filepath = "/path/to/a/file.123sdufg.sdfs.tiff"
        true_basename = "file.123sdufg.sdfs"
        basename = get_basename(filepath)
        self.assertEqual(basename, true_basename)

        filepath = "/path/to/a/file"
        true_basename = "file"
        basename = get_basename(filepath)
        self.assertEqual(basename, true_basename)

    def test_get_dtype(self):
        types = [
            (1, False, False, np.uint8),
            (2, False, False, np.uint16),
            (2, False, True, np.int16),
            (4, False, True, np.float32),
            (8, False, True, np.float64),
            (8, True, True, np.complex64),
            (16, True, True, np.complex128),
        ]

        for depth, is_complex, signed, dtype in types:
            self.assertEqual(get_dtype(depth, is_complex, signed), dtype)

        with self.assertRaises(AssertionError):
            get_dtype(1, True, True)

    def test_get_gdal_dtype(self):

        types = [
            (1, False, False, gdal.GDT_Byte),
            (2, False, False, gdal.GDT_UInt16),
            (2, False, True, gdal.GDT_Int16),
            (4, False, True, gdal.GDT_Float32),
            (8, False, True, gdal.GDT_Float64),
            (8, True, True, gdal.GDT_CFloat32),
            (16, True, True, gdal.GDT_CFloat64),
        ]

        for depth, is_complex, signed, dtype in types:
            self.assertEqual(get_gdal_dtype(depth, is_complex, signed), dtype)

        with self.assertRaises(AssertionError):
            get_gdal_dtype(1, True, True)

    def test_gdal_to_numpy_datatype(self):

        types = [
            (gdal.GDT_Byte, np.uint8),
            (gdal.GDT_Int16, np.int16),
            (gdal.GDT_UInt16, np.uint16),
            (gdal.GDT_UInt32, np.uint32),
            (gdal.GDT_Int32, np.int32),
            (gdal.GDT_Float32, np.float32),
            (gdal.GDT_Float64, np.float64),
            (gdal.GDT_CInt16, np.complex64),
            (gdal.GDT_CInt32, np.complex64),
            (gdal.GDT_CFloat32, np.complex64),
            (gdal.GDT_CFloat64, np.complex128),
        ]

        for gdal_type, dtype in types:
            self.assertEqual(gdal_to_numpy_datatype(gdal_type), dtype)

        with self.assertRaises(AssertionError):
            gdal_to_numpy_datatype(1000)


if __name__ == "__main__":

    suite = TestLoader().loadTestsFromTestCase(TestCommon)
    TextTestRunner().run(suite)

