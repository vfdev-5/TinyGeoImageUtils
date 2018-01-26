
from __future__ import absolute_import

import os
import numpy as np
from glob import glob
import logging

import gdal
import osr

from ..common import numpy_to_gdal_datatype


logger = logging.getLogger('gimg')


# Generate map between extensions and gdal drivers:
EXTENSIONS_GDAL_DRIVER_CODE_MAP = {}
for i in range(gdal.GetDriverCount()):
    drv = gdal.GetDriver(i)
    if drv.GetMetadataItem(gdal.DCAP_RASTER):
        ext = drv.GetMetadataItem(gdal.DMD_EXTENSION)
        if ext is not None:
            EXTENSIONS_GDAL_DRIVER_CODE_MAP[drv.GetMetadataItem(gdal.DMD_EXTENSION)] = drv.ShortName


def get_files_from_folder(input_dir, extensions=None):
    output = []
    if extensions is None:
        extensions = [""]

    for ext in extensions:
        output.extend(glob(os.path.join(input_dir, "**", "*" + ext), recursive=True))

    return output


def write_to_file(data, output_filepath, geo_info=None, metadata=None):
    """
    Method to write ndarray on disk using GDAL
    :param data: ndarray of shape (h, w, c)
    :param output_filepath: filepath where to write the data
    :param geo_info: Dictionary with keys, values: 'epsg': epsg_code,
        'geo_extent': [top-left, top-right, bottom-right, bottom-left]
    :param metadata: (dict) additional metadata
    """
    assert isinstance(data, np.ndarray) and len(data.shape) == 3, "Input data should be ndarray of shape (h, w, c)"
    assert not os.path.exists(output_filepath), "Output filepath '%s' already exists" % output_filepath

    if geo_info is not None:
        assert isinstance(geo_info, dict) and "epsg" in geo_info and "geo_extent" in geo_info, \
            "Argument geo_info should be a dictionary with keys, values: " + \
            "'epsg': epsg_code, 'geo_transform': [a, b, c, d, e, f]"

    if metadata is not None:
        assert isinstance(metadata, dict), "Metadata argument should be a dictionary"

    ext = os.path.basename(output_filepath).split(os.path.extsep)
    ext = ext[-1] if len(ext) > 0 else None
    if ext is None or ext not in EXTENSIONS_GDAL_DRIVER_CODE_MAP:
        logger.warning("GDAL does not support writing files with extension: %s" % ext)
        driver_code = "GTiff"
        output_filepath += ".tif"
    else:
        driver_code = EXTENSIONS_GDAL_DRIVER_CODE_MAP[ext]
    driver = gdal.GetDriverByName(driver_code)
    assert driver is not None, "Failed to create a driver to write the file. Extension: %s, Driver code: %s" \
                               % (ext, driver_code)

    gdal_dtype = numpy_to_gdal_datatype(data.dtype)
    assert gdal_dtype is not None, "Failed to convert input data type {} into gdal data type".format(data.dtype)

    height, width, n_bands = data.shape
    dst_ds = driver.Create(output_filepath, width, height, n_bands, gdal_dtype)
    assert dst_ds is not None, "Failed to write file %s" % output_filepath

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(geo_info['epsg'])
    proj = sr.ExportToWkt()
    dst_ds.SetProjection(proj)

    geo_transform = compute_geo_transform(geo_info['geo_extent'])
    dst_ds.SetGeoTransform(geo_transform)

    dst_ds.SetMetadata(metadata)

    for i in range(1, n_bands + 1):
        band_data = data[:, :, i - 1]
        dst_band = dst_ds.GetRasterBand(i)
        dst_band.WriteArray(band_data, 0, 0)

    dst_ds = None

