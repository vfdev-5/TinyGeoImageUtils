from __future__ import absolute_import

import os

import numpy as np

import gdal
import osr


# Project
from gimg.common import get_gdal_dtype, get_dtype
from gimg.GeoImage import compute_geo_extent

"""
    Script to create synthetic images
"""


def create(width, height, nb_bands, filepath, depth=2, is_complex=False, metadata=None,
           geo_transform=(13.60746033, 0.001, 0.0, 50.25013288, 0.0, -0.001),
           epsg=4326):
    """
        Write a synthetic image
    """
    # Write a small test image
    data = np.zeros((height, width, nb_bands), dtype=get_dtype(depth, is_complex))
    step_h = height//10
    step_w = width//10
    for i in range(0, height, step_h):
        for j in range(0, width, step_w):
            data[i:i+step_h, j:j+step_w, :] += np.random.randint(0, 255, size=(1, 1, nb_bands), dtype=np.uint16)

    driver = gdal.GetDriverByName('GTiff')
    dt = get_gdal_dtype(depth, is_complex)
    ds = driver.Create(filepath, width, height, nb_bands, dt)
    for i in range(0, nb_bands):
        ds.GetRasterBand(i+1).WriteArray(data[:, :, i])

    # Add metadata
    if metadata is None:
        ds.SetMetadata({'TEST0': '0', 'TEST1': '123', 'TEST2': 'abc'})
    else:
        ds.SetMetadata(metadata)

    # GeoTransform = [x, dx, dy, y, dx, dy]
    if geo_transform is not None:
        ds.SetGeoTransform(geo_transform)

    if epsg is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())

    src = None
    ds = None
    driver = None
    return data


def create_synthetic_image_file(local_temp_folder, shape, depth, is_complex):
    # Create local synthetic image:
    filepath = os.path.join(local_temp_folder, 'test_small_image.tif')
    metadata = {'key_1': 'value_1', 'key_2': "1 2 3", 'key_3': '3'}
    geo_transform = (13.60746033, 0.001, 0.0005, 50.25013288, 0.0005, -0.001)
    geo_extent = compute_geo_extent(geo_transform, shape)
    epsg = 4326
    data = create(shape[1], shape[0], shape[2], filepath,
                  depth=depth, is_complex=is_complex,
                  metadata=metadata, geo_transform=geo_transform, epsg=epsg)
    return filepath, data, geo_extent, metadata, geo_transform, epsg


def create_virt_image(w, h, c, dtype):
    # Create a synthetic gdal dataset
    data = np.arange(0, w*h*c, dtype=dtype).reshape((h, w, c))
    driver = gdal.GetDriverByName('MEM')
    gdal_dtype = get_gdal_dtype(data[0, 0, 0].itemsize,
                                data[0, 0, 0].dtype == np.complex64 or
                                data[0, 0, 0].dtype == np.complex128,
                                signed=False if dtype in (np.uint8, np.uint16) else True)
    ds = driver.Create('', w, h, c, gdal_dtype)
    for i in range(0, c):
        ds.GetRasterBand(i+1).WriteArray(data[:, :, i])
    return ds, data