# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:01:46 2023

@author: Hadi
"""

import cv2
import numpy as np
import yuvio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fftpack import dct

# Set the path to the YUV file
file_path = r'akiyo_cif.y4m'

# Set the video resolution and frame rate
width = 352
height = 288
fps = 30

# Open the YUV file
yuv_file = open(file_path, 'rb')

# Create the output file for the DCT frames
output_file_path = r'akiyo_cif_dct.yuv'
output_file = open(output_file_path, 'wb')

# Calculate the size of a single frame in bytes
frame_size = int(width * height * 1.5)

# Read the first YUV frame
yuv_frame = yuvio.imread(file_path, width, height, "yuv420p")

# Extract the Y component from the YUV frame
y_frame = yuv_frame.y

# Perform 2D DCT on the Y component
dct_frame = dct(dct(y_frame, axis=0, norm='ortho'), axis=1, norm='ortho')

# Display the original and DCT frames
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(y_frame, cmap='gray')
axs[0].set_title('Original Frame')
axs[1].imshow(dct_frame, cmap='gray')
axs[1].set_title('DCT Frame')
plt.show()
