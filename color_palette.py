import imageio
from sklearn.cluster import KMeans
import argparse
import numpy as np

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
	print(img.dtype)

	# Set up kmeans init so that there's hopefully a cluster
	# for each luminance level
	if args.random_colors:
		inits = 'random'
	else:
		lum = np.mean(img, axis=-1).flatten()
		min_lum = np.min(lum)
		max_lum = np.max(lum)
		inits = np.linspace(min_lum, max_lum, args.num_colors, dtype=int)
		inits = np.stack([inits, inits, inits], axis=1)
		print(inits)
		assert inits.shape == (args.num_colors, 3), f'Bad shape! {inits.shape}'

	# Get colors
	kmeans = KMeans(
		n_clusters=args.num_colors,
		init=inits,
		n_init=1,
		verbose=1,
	).fit(img.reshape(i_H * i_W, i_C))

	# Sort colors darkest to brightest
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
