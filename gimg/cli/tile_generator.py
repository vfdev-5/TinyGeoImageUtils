from __future__ import absolute_import

import os
import click

from ..GeoImageTilers import GeoImageTiler, GeoImageTilerConstSize


def get_files_from_folder(input_dir, extensions):
    return []



@click.group()
def cli():
    pass

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
@click.argument('tile_size', type=int, help="Output tile size in pixels")
@click.argument('min_overlapping', type=list, help="Minimal overlapping between tiles in pixels")
@click.option('--extensions', type=list, default=(), help="List of file extensions to select")
@click.option('--output_extension', type=list, default="tif", help="Output tile file extension")
@click.option('--n_workers', default=4, help="Number of workers in the processing pool [default=4]")
def run_const_size_tiler(input_dir, output_dir, tile_size, min_overlapping, extensions, output_extensions, n_workers):
    files = get_files_from_folder(input_dir, extensions)
    pass


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
