import os
import logging

# start logging
logging.info("Start BicycleGAN")
logger = logging.getLogger('BicycleGAN')
logger.setLevel(logging.INFO)

def makedirs(path):
    if not os.path.exists(path):
		os.makedirs(path)
