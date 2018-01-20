import numpy as np
import os
import h5py
from glob import glob
import scipy.misc
from scipy.misc import imread, imresize

def load_images(dir):
	all_images = []
	for sub_dir in ['train', 'val']:
		folder_path = os.path.join(dir, sub_dir)
		all_img_A = []
		all_img_B = []
		for path in glob(os.path.join(folder_path, '*.jpg')):
			full_image = imread(path)
			img_A = full_image[:, :full_image.shape[1]/2, :]
			img_B = full_image[:, full_image.shape[1]/2:, :]
			all_img_A.append(img_A)
			all_img_B.append(img_B)

		#all_images[(sub_dir+'A')] = img_A
		#all_images[(sub_dir+'B')] = img_B
		all_images.append((sub_dir+'A', all_img_A))
		all_images.append((sub_dir+'B', all_img_B))

	all_images = load_images('dir')
	train_A = []
	train_B = []
	val_A = []
	val_B = []

	train_A = np.asarray(all_images[0][1])
	train_B = np.asarray(all_images[1][1])
	val_A = np.asarray(all_images[2][1])
	val_B = np.asarray(all_images[3][1])

	return train_A, train_B, val_A, val_B

def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)