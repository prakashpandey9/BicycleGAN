import numpy as np
import os
import h5py
import glob
import scipy.misc
from scipy.misc import imread, imresize

def load_images():
	train_all=glob.glob("cityscapes/train/*.jpg")
	train_img_A = []
	train_img_B = []

	for file in train_all:
		full_image = imread(file)
		img_A = full_image[:, :full_image.shape[1]/2, :]
	 	img_B = full_image[:, full_image.shape[1]/2:, :]
	 	train_img_A.append(img_A)
	 	train_img_B.append(img_B)

	train_A = np.asarray(train_img_A)
	train_B = np.asarray(train_img_B)
	print train_A.shape
	print train_B.shape

	test_all = glob.glob("cityscapes/val/*.jpg")
	test_img_A = []
	test_img_B = []

	for file in test_all:
		full_image = imread(file)
		img_A = full_image[:, :full_image.shape[1]/2, :]
	 	img_B = full_image[:, full_image.shape[1]/2:, :]
	 	test_img_A.append(img_A)
	 	test_img_B.append(img_B)

	test_A = np.asarray(test_img_A)
	test_B = np.asarray(test_img_B)
	print test_A.shape
	print test_B.shape
	
	return train_A, train_B, test_A, test_B

def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)

def inverse_transform(images):
	return (images+1.)/2.
