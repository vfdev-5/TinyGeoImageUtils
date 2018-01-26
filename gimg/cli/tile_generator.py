from __future__ import absolute_import

import os

from multiprocessing import Pool

import numpy as np

import click


from ..common import get_basename
from ..GeoImage import GeoImage
from ..GeoImageTilers import GeoImageTiler, GeoImageTilerConstSize
from . import get_files_from_folder, write_to_file


@click.group()
def cli():
    pass


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
@click.argument('tile_size_in_pixels', type=int)
@click.argument('min_overlapping_in_pixels', type=list)
@click.option('--extensions', type=list, default=None, help="List of file extensions to select")
@click.option('--output_extension', type=list, default="tif", help="Output tile file extension")
@click.option('--n_workers', default=4, type=int, help="Number of workers in the processing pool [default=4]")
def run_const_size_tiler(input_dir, output_dir, tile_size, min_overlapping, extensions, output_extension, n_workers):
    files = get_files_from_folder(input_dir, extensions)

    def get_task(output_dir, tile_size, min_overlapping, output_extension):
        def run_task(filepath):

            output_tiles_dir = os.path.join(output_dir, get_basename(filepath) + "_tiles")
            os.makedirs(output_tiles_dir)

            geo_image = GeoImage(filepath)
            tiles = GeoImageTilerConstSize(geo_image, tile_size=tile_size, min_overlapping=min_overlapping)
            for tile, x, y in tiles:

                # transform 4 image corners
                h, w = tile.shape[0], tile.shape[1]
                pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                geo_extent = geo_image.transform(pts, "pix2geo")

                geo_info = {
                    'epsg': geo_image.get_epsg(),
                    'geo_extent': geo_extent
                }
                metadata = geo_image.metadata
                output_tile_filepath = os.path.join(output_tiles_dir, "tile_%i_%i.%s" % (x, y, output_extension))
                write_to_file(tile, output_tile_filepath, geo_info, metadata)

        return run_task

    with Pool(n_workers) as pool:
        pool.map(get_task(output_dir, tile_size, min_overlapping, output_extension), files)


cli.add_command(run_const_size_tiler, name="const_size")


# @click.command()
# @click.argument('input_dir', type=click.Path(exists=True))
# @click.argument('output_dir', type=click.Path(exists=True))
# @click.argument('overlapping', type=int, help="Overlapping between tiles in pixels")
# @click.option('--extensions', type=list, default=(), help="List of file extensions to select")
# def run_var_size_tiler(input_dir, output_dir, tile_size, extensions):
#     pass
#
# cli.add_command(run_var_size_tiler, name="const_size")

if __name__ == "__main__":
    cli()