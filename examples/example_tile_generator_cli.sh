#!/usr/bin/env bash

#
# Run tile generator on dog.jpg and create overlapping const size tiles of size 256x256
#
#        Usage: tile_generator.py const_size [OPTIONS] INPUT_DIR_OR_FILE OUTPUT_DIR
#                                            TILE_SIZE_IN_PIXELS
#                                            MIN_OVERLAPPING_IN_PIXELS
#
#        Options:
#          --extensions TEXT        String of file extensions to select (if input is a
#                                   directory), e.g. 'jpg,png,tif'
#          --output_extension TEXT  Output tile file extension
#          --n_workers INTEGER      Number of workers in the processing pool
#                                   [default=4]
#          -q, --quiet              Disable verbose mode
#          --help                   Show this message and exit.

mkdir tiles

tile_generator const_size --extensions="jpg,png" . tiles 256 20
