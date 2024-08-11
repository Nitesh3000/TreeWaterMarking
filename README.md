# Tree Ring Watermarking Project
## Overview
This project implements a watermarking technique in diffusion models using wavelet transforms. The core contributions include adding watermarking functionality in both the highest and lowest frequency components of images using Dual-Tree Complex Wavelet Transform (DTCWT). The project also provides comprehensive image generation and visualization tools.

## Features
### Dual-Tree Complex Wavelet Transform (DTCWT): 
  Implements both forward and inverse DTCWT to perform watermark embedding and extraction.
### Watermarking in Frequency Components: 
Watermarks are embedded in both high and low-frequency components to ensure durability against various attacks.
### Flexible Watermark Patterns: 
Supports various watermark patterns including random, zero-filled, constant, and ring-shaped watermarks.
### Image Processing and Visualization: 
Includes utilities for saving and visualizing watermarked images and their transformed components.

## Contribution
### Wavelet Transform Integration: 
Added support for watermarking in both high and low frequency components using Dual-Tree Complex Wavelet Transform (DTCWT). This allows for more robust and fine-grained control over watermark placement in image frequency domains.
### Image Watermarking: 
Implemented watermark injection in the highest and lowest frequency components, ensuring the watermark is embedded effectively across different scales of the image.
### Visualization: 
Developed functionality to save various stages of image processing for visualization purposes, including before and after watermarking, as well as the effect of watermarking in the frequency domain.

## Folder Structure
### output_images/
This folder contains subfolders where images are saved for each image processed. The saved images include:

latent_no_w_image.png: Latent image before watermarking.

latent_w_image.png: Latent image after watermarking.

dwt_no_w_image.png: DWT image before watermarking (real part of the low-frequency component).

dwt_w_image.png: DWT image after watermarking (real part of the modified low-frequency component).

yh_w_image_highest.png: Real part of the highest frequency components after watermarking.

Other Images: Various other visualizations to assist in evaluating the watermarking process.

