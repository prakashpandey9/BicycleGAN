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

def save_image(image, size, img_path):
	return imsave(inverse_transform(image), size, img_path)

def imsave(image, img_size, img_path):
	image = np.squeeze(merge(image, img_size))
	return scipy.misc.imsave(img_path, image)

def inverse_transform(image):
	return (image+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
	raise ValueError('In merge function, the first argument must have dimensions: HxW or HxWx3 or HxWx4')
