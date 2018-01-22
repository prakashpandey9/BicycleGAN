from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from model import BicycleGAN
from folder import check_folder
#from load_data import load_images


def parse_args():
	desc = "Tensorflow implementation of BicycleGAN"
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--Z_dim', type=int, default=8, help='Size of latent vector')
	parser.add_argument('--reconst_coeff', type=float, default=10, help='Reconstruction Coefficient')
	parser.add_argument('--latent_coeff', type=float, default=0.5, help='Latent Coefficient')
	parser.add_argument('--kl_coeff', type=float, default=0.01, help='KL Coefficient')
	parser.add_argument('--num_epochs', type=int, default=10, help='Total number of epochs')
	parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning Rate')
	parser.add_argument('--image_size', type=int, default=256, help='Image Size')
	parser.add_argument('--batch_size', type=int, default=1, help='Size of the minibatch')
	parser.add_argument('--gan_type', type=str, default='BicycleGAN', help='Type of GAN')
	parser.add_argument('--dataset', type=str, default='cityscapes', help='The name of dataset')
	parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',help='Directory name to save the checkpoints')
	parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
	parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
	return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
	# --checkpoint_dir
	check_folder(args.checkpoint_dir)

	# --result_dir
	check_folder(args.result_dir)

	# --result_dir
	check_folder(args.log_dir)

	# --epoch
	assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

	# --batch_size
	assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

	# --z_dim
	assert args.Z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

	return args

"""main"""
def main():
	# parse arguments
	args = parse_args()
	if args is None:
	  exit()

	# open session
	model = BicycleGAN
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	    # declare instance for GAN

	    gan = None
	    if args.gan_type == model.model_name:
	        gan = model(sess,
	                    epoch=args.epoch,
	                    batch_size=args.batch_size,
	                    Z_dim=args.Z_dim,
	                    image_size=args.image_size,
	                    dataset_name=args.dataset,
	                    checkpoint_dir=args.checkpoint_dir,
	                    result_dir=args.result_dir,
	                    log_dir=args.log_dir)
	    if gan is None:
	        raise Exception("[!] There is no option for " + args.gan_type)

	    # build graph
	    gan.build_model()

	    # show network architecture
	    # show_all_variables()

	    # launch the graph in a session
	    gan.train()
	    print(" [*] Training finished!")

	    # visualize learned generator
	    gan.test()
	    print(" [*] Testing finished!")

if __name__ == '__main__':
	main()
