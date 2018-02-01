# Multimodal Image-to-Image Translation
This is a Tensorflow implementation of the NIPS paper "Toward Multimodal Image-to-Image Translation". The aim is to generate a distribution of output images given an input image. Basically, it is an extension of image to image translation model using Conditional Generative Adversarial Networks.

The idea is to learn a low-dimensional latent representation of target images using an encoder net i.e., a probability distribution which has generated all the target images and to learn the joint probability distribution of this latent vector as P(z). In this model, the mapping from latent vector to output images and output image to latent vector is bijective. The overall architecture consists of two cycle, B->z->B' and z->B'->z' and hence the name BicycleGAN.

![Model Architecture](https://github.com/prakashpandey9/BicycleGAN/blob/master/cityscapes/figures/BicycleGAN.png)

Image Source : [Toward Multimodal Image-to-Image Translation][1] Paper

## Description
- We have 3 different networks: a) Discriminator, b) Encoder, and c) Generator
- A cGAN-VAE (Conditional Generative Adversarial Network- Variational Autoencoder) is used to encode the ground truth output image B to latent vector z which is then used to reconstruct the output image B' i.e., B -> z -> B'
- For inverse mapping (z->B'->z'), we use LR-GAN (Latent Regressor Generative Adversarial Networks) in which a Generator is used to generate B' from input image A and z.
- Combining both these models, we get BicycleGAN.
- The architecture of Generator is same as U-net in which there are encoder and decoder nets with symmetric skip connections.
- For Encoder, we use several residual blocks for an efficient encoding of the input image.
- The model is trained using Adam optimizer using BatchNormalization with batch size 1.
- LReLU activation function is used for all types of networks.

## Requirements
- Python 2.7
- Numpy
- Tensorflow
- Scipy

## Training / Testing
After cloning this repository, you can train the network by running the following command.
```shell
$ mkdir test_results
$ python main.py
```

## References
- Toward Multimodal Image-to-Image Translation[2] Paper
- pix2pix Paper[3] Paper

## License
MIT

[1]:https://arxiv.org/pdf/1711.11586.pdf
[2]:https://arxiv.org/pdf/1711.11586.pdf
[3]:https://arxiv.org/pdf/1611.07004.pdf
