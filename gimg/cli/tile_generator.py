from __future__ import absolute_import

import os
from functools import partial

from multiprocessing import Pool

import numpy as np

import click


from ..common import get_basename
from ..GeoImage import GeoImage
from ..GeoImageTilers import GeoImageTilerConstSize
from . import get_files_from_folder, write_to_file


@click.group()
def cli():
    pass


def run_const_size_task(filepath, output_dir, tile_size, min_overlapping, output_extension):
    output_tiles_dir = os.path.join(output_dir, get_basename(filepath) + "_tiles")
    os.makedirs(output_tiles_dir)
    geo_image = GeoImage(filepath)
    tiles = GeoImageTilerConstSize(geo_image, tile_size=(tile_size, tile_size), min_overlapping=min_overlapping)
    for tile, x, y in tiles:
        geo_info = None
        metadata = None
        if geo_image.projection is not None and len(geo_image.projection) > 0:
            # transform 4 image corners
            h, w = tile.shape[0], tile.shape[1]
            pts = np.array([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]])
            geo_extent = geo_image.transform(pts, "pix2proj")
            geo_info = {
                'epsg': geo_image.get_epsg(),
                'geo_extent': geo_extent
            }
        if geo_image.metadata is not None:
            metadata = geo_image.metadata
        output_tile_filepath = os.path.join(output_tiles_dir, "tile_%i_%i.%s" % (x, y, output_extension))
        write_to_file(tile, output_tile_filepath, geo_info, metadata)


@click.command()
@click.argument('input_dir_or_file', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument('output_dir', type=click.Path(exists=True))
@click.argument('tile_size_in_pixels', type=int)
@click.argument('min_overlapping_in_pixels', type=int)
@click.option('--extensions', type=str, default=None,
              help="String of file extensions to select (if input is a directory), e.g. 'jpg,png,tif'")
@click.option('--output_extension', type=str, default="tif", help="Output tile file extension")
@click.option('--scale', type=float, help="Scale input before tiling")
@click.option('--n_workers', default=4, type=int, help="Number of workers in the processing pool [default=4]")
@click.option('-q', '--quiet', is_flag=True, help='Disable verbose mode')
def run_const_size_tiler(input_dir_or_file, output_dir, tile_size_in_pixels, min_overlapping_in_pixels, extensions,
                         output_extension, scale, n_workers, quiet):

    if os.path.isdir(input_dir_or_file):
        if extensions is not None:
            extensions = extensions.split(",")
        files = get_files_from_folder(input_dir_or_file, extensions)
        assert len(files) > 0, "No files with extensions '{}' found at '{}'".format(extensions, input_dir_or_file)
    else:
        files = [input_dir_or_file]

    chunk_size = 10
    if n_workers > 1 and len(files) > chunk_size // 2:
        with Pool(n_workers) as pool:
            for i in range(0, len(files), chunk_size):
                if not quiet:
                    print("%i%%" % int(i * 100.0 / len(files)), end=" . ")
                chunk_files = files[i: i + chunk_size]
                pool.map(partial(run_const_size_task,
                                 output_dir=output_dir,
                                 tile_size=tile_size_in_pixels,
                                 min_overlapping=min_overlapping_in_pixels,
                                 output_extension=output_extension), chunk_files)
    else:
        for i, f in enumerate(files):
            if not quiet:
                print("%i%%" % int(i * 100.0 / len(files)), end=" . ")
            run_const_size_task(f, output_dir=output_dir, tile_size=tile_size_in_pixels,
                                min_overlapping=min_overlapping_in_pixels, output_extension=output_extension)
    if not quiet:
        print("100%")


cli.add_command(run_const_size_tiler, name="const_size")


if __name__ == "__main__":
    cli()
