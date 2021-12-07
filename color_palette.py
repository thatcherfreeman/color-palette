import imageio
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import argparse
import numpy as np
import colorsys

def subsample_points_by_space(points, side_dim):
	# Make KD-Tree from points
	nn = KDTree(points)
	# Create num_samples query points evenly spaced across the RGB cube
	max_val = np.max(points)
	min_val = np.min(points)
	side = np.linspace(min_val, max_val, side_dim)
	interval = (max_val - min_val) / side_dim
	lattice = cartesian_product(*[side, side, side])
	# Sample nearest point in the image for each point in the lattice
	d, i = nn.query(lattice, k=1, return_distance=True)
	output_samples = points[i[d < interval]]
	return output_samples

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

if __name__ == '__main__':
	args = argparse.ArgumentParser()
	args.add_argument(
		'--file',
		type=str,
		help='Specify the filename of the image you want to generate a color palette for.',
	)
	args.add_argument(
		'--num_colors',
		type=int,
		default=10,
		help='Number of colors to be included in the color palette.',
	)
	args.add_argument(
		'--margin',
		type=float,
		default=0.1,
		help='size of the white margin around the color palette (as a fraction of the size of the color chip',
	)
	args.add_argument(
		'--out_file',
		type=str,
		default=None,
		help='Specify to override the output filepath.',
	)
	args.add_argument(
		'--random_colors',
		action='store_true',
		help='Use this flag to initialize kmeans with random colors',
	)
	args.add_argument(
		'--sample_extreme_colors',
		action='store_true',
		help='Experimental method that tries to subsample the input image as evenly as possible within the RGB cube, therefore removing duplicate colors.'
	)
	args.add_argument(
		'--hsv_initialization',
		action='store_true',
		help='Initialize KMeans using colors distributed by hue rather than by luminance',
	)
	args = args.parse_args()
	fn = args.file
	if args.out_file is None:
		out_fn = f'{fn[:-4]}_palatte.jpg'
	else:
		out_fn = args.out_file

	# Read image
	img = imageio.imread(fn)
	if img.shape[-1] == 4:
		img = img[:, :, :3]
	assert img.shape[-1] == 3, f'Bad shape! {img.shape}'
	i_H, i_W, i_C = img.shape
	o_H, o_W, o_C = int(i_H*1.2), i_W, i_C
	print(f'Image format: {img.dtype}')

	# Set up kmeans init so that there's hopefully a cluster
	# for each luminance level
	lum = np.mean(img, axis=-1).flatten()
	min_lum = np.min(lum)
	max_lum = np.max(lum)
	if args.random_colors:
		inits = 'random'
	elif args.hsv_initialization:
		hues = np.linspace(0, 1.0, args.num_colors)
		inits = [list(colorsys.hsv_to_rgb(h, 1.0, 1.0)) for h in hues]
		inits = np.array(inits) * max_lum
		assert inits.shape == (args.num_colors, 3), f'Bad shape! {inits.shape}'
	else:
		inits = np.linspace(min_lum, max_lum, args.num_colors, dtype=int)
		inits = np.stack([inits, inits, inits], axis=1)
		assert inits.shape == (args.num_colors, 3), f'Bad shape! {inits.shape}'

	print('Sampling points....')
	samples = img.reshape(i_H * i_W, i_C)
	samples = samples[np.random.choice(samples.shape[0], 200000), :]
	if args.sample_extreme_colors:
		samples = subsample_points_by_space(samples, 16)

	# Get colors
	print(f'Getting {args.num_colors} best colors...')
	kmeans = KMeans(
		n_clusters=args.num_colors,
		init=inits,
		n_init=1,
	).fit(samples)

	# Sort colors darkest to brightest
	print(f'Drawing image....')
	palatte = kmeans.cluster_centers_
	sort_order = np.argsort(np.mean(palatte, axis=-1))
	palatte = palatte[sort_order, :]
	assert palatte.shape == (args.num_colors, 3)

	output_img = np.full((o_H, o_W, o_C), 255)
	output_img[:i_H, :i_W, :i_C] = img[:, :, :]

	color_width = i_W / args.num_colors
	margin = color_width * args.margin
	for i, c in enumerate(palatte):
		output_img[ \
			int(i_H+margin):int(o_H-margin),\
			int(i*color_width+margin):int((i+1)*color_width-margin),\
			:
		] = c
	output_img = np.clip(output_img, 0, np.max(img)).astype(np.uint8)
	print(f'Writing output to {out_fn}')
	imageio.imwrite(out_fn, output_img)
