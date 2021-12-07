# Color Palette Generator

Generates a color palette for a given image and adds it to the bottom of the frame. The colors are selected using a K-Means clustering algorithm.

## Installation instructions

1. Install a recent version of python.
2. Clone the repository somewhere
3. Run the command `pip install -r requirements.txt` in the terminal

## Running instructions

Run the following command:

```
python color_palette.py --file path/to/file.jpg
```

This will generate a new file in the same folder as your original image, but with the word `_palette` appended to the file name. This script should work with whatever file extensiosn that the imageio library supports. Currently the code expects 8-bit files.

You can optionally use the following arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Specify the filename of the image you want to generate a color palette for.
  --num_colors NUM_COLORS
                        Number of colors to be included in the color palette.
  --margin MARGIN       size of the white margin around the color palette (as a fraction of the size of the color chip
  --out_file OUT_FILE   Specify to override the output filepath.
  --random_colors       Use this flag to initialize kmeans with random colors
```