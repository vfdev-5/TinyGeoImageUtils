from __future__ import absolute_import

import os
import click

from ..GeoImageTilers import GeoImageTiler, GeoImageTilerConstSize


@click.group()
def cli():
    pass

# @click.command()
# @click.argument('input_dir', type=click.Path(exists=True))
# @click.argument('output_dir', type=click.Path(exists=True))
# @click.argument('tile_size', type=int, help="Output tile size in pixels")
# @click.option('--extensions', type=list, default=(), help="List of file extensions to select")
# def run_const_size_tiler(input_dir, output_dir, tile_size, extensions):
#     pass
#
#
# cli.add_command(run_const_size_tiler, name="const_size")
#
#
# @click.command()
# @click.argument('input_dir', type=click.Path(exists=True))
# @click.argument('output_dir', type=click.Path(exists=True))
# @click.argument('overlapping', type=int, help="Overlapping between tiles in pixels")
# @click.option('--extensions', type=list, default=(), help="List of file extensions to select")
# def run_var_size_tiler(input_dir, output_dir, tile_size, extensions):
#     pass
#
# cli.add_command(run_const_size_tiler, name="const_size")


if __name__ == "__main__":
    cli()
