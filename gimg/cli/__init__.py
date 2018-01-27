
import os
import numpy as np
from glob import glob
import logging

import gdal
import osr

from ..common import numpy_to_gdal_datatype
from ..GeoImage import compute_geo_transform


logger = logging.getLogger('gimg')


# Generate map between extensions and gdal drivers:
def _create_ext_driver_code_map():
    if hasattr(gdal, "DCAP_RASTER"):
        def _check_driver(drv):
            return drv.GetMetadataItem(gdal.DCAP_RASTER)
    else:
        def _check_driver(drv):
            return True
    output = {}
    for i in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(i)
        if _check_driver(drv):
            if drv.GetMetadataItem(gdal.DCAP_CREATE) or drv.GetMetadataItem(gdal.DCAP_CREATECOPY):
                ext = drv.GetMetadataItem(gdal.DMD_EXTENSION)
                if ext is not None and len(ext) > 0:
                    output[drv.GetMetadataItem(gdal.DMD_EXTENSION)] = drv.ShortName
    return output


EXTENSIONS_GDAL_DRIVER_CODE_MAP = _create_ext_driver_code_map()


def get_files_from_folder(input_dir, extensions=None):
    output = []
    if extensions is None:
        extensions = [""]

    for ext in extensions:
        output.extend(glob(os.path.join(input_dir, "**", "*" + ext), recursive=True))

    return output


def write_to_file(data, output_filepath, geo_info=None, metadata=None, options=None):
    """
    Method to write ndarray on disk using GDAL
    :param data: ndarray of shape (h, w, c)
    :param output_filepath: filepath where to write the data
    :param geo_info: Dictionary with keys, values: 'epsg': epsg_code,
        'geo_extent': [top-left, top-right, bottom-right, bottom-left]
    :param metadata: (dict) additional metadata
    :param options: options argument passed to gdal.Driver.Create() method.
        For example, options = ['COMPRESS=LZW']
    """
    assert isinstance(data, np.ndarray) and len(data.shape) == 3, "Input data should be ndarray of shape (h, w, c)"
    assert not os.path.exists(output_filepath), "Output filepath '%s' already exists" % output_filepath

    if geo_info is not None:
        assert isinstance(geo_info, dict) and "epsg" in geo_info and "geo_extent" in geo_info, \
            "Argument geo_info should be a dictionary with keys, values: " + \
            "'epsg': epsg_code, 'geo_extent': [top-left, top-right, bottom-right, bottom-left]"

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

    # Test if driver can create or we need to create a copy
    driver = gdal.GetDriverByName(driver_code)
    assert driver is not None, "Failed to create a driver to write the file. Extension: %s, Driver code: %s" \
                               % (ext, driver_code)

    gdal_dtype = numpy_to_gdal_datatype(data.dtype)
    assert gdal_dtype is not None, "Failed to convert input data type {} into gdal data type".format(data.dtype)

    height, width, n_bands = data.shape
    kwargs = {}
    if options is not None:
        kwargs['options'] = options

    def _create_dataset(drv, output_filepath, width, height, n_bands, gdal_dtype, **kwargs):
        if drv.GetMetadataItem(gdal.DCAP_CREATE):
            return drv.Create(output_filepath, width, height, n_bands, gdal_dtype, **kwargs), False
        else:
            mem_drv = gdal.GetDriverByName('MEM')
            return mem_drv.Create('', width, height, n_bands, gdal_dtype), True

    dst_ds, need_finalize = _create_dataset(driver, output_filepath, width, height, n_bands, gdal_dtype, **kwargs)
    assert dst_ds is not None, "Failed to write file %s" % output_filepath

    if geo_info is not None:
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(geo_info['epsg'])
        proj = sr.ExportToWkt()
        dst_ds.SetProjection(proj)

        geo_transform = compute_geo_transform(geo_info['geo_extent'], (height, width))
        dst_ds.SetGeoTransform(geo_transform)

    if metadata is not None:
        dst_ds.SetMetadata(metadata)

    for i in range(1, n_bands + 1):
        band_data = data[:, :, i - 1]
        dst_band = dst_ds.GetRasterBand(i)
        dst_band.WriteArray(band_data, 0, 0)

    if need_finalize:
        driver.CreateCopy(output_filepath, dst_ds, False, **kwargs)
