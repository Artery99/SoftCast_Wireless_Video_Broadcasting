# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:15:30 2023

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

# Set the number of frames in the GoP
gop_size = 5

# Open the YUV file
yuv_file = open(file_path, 'rb')

# Calculate the size of a single frame in bytes
frame_size = int(width * height * 1.5)

# Read the first YUV frame
yuv_frame = yuvio.imread(file_path, width, height, "yuv420p")

# Extract the Y component from the YUV frame
y_frame = yuv_frame.y

# Initialize the GoP array
gop = np.empty((gop_size, height, width))

# Set the first frame in the GoP to the extracted Y component
gop[0] = y_frame

# Read the next (gop_size - 1) frames into the GoP array
for i in range(1, gop_size):
    # Seek to the next frame in the YUV file
    yuv_file.seek(i * frame_size)

    # Read the YUV frame
    yuv_frame = yuvio.imread(yuv_file, width, height, "yuv420p")

    # Extract the Y component from the YUV frame and add it to the GoP array
    gop[i] = yuv_frame.y

# Perform 3D DCT on the GoP
dct_gop = dct(dct(dct(gop, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')

# Display the original and DCT GoP
fig, axs = plt.subplots(2, gop_size, figsize=(20, 5))
for i in range(gop_size):
    axs[0, i].imshow(gop[i], cmap='gray')
    axs[0, i].set_title('Frame {}'.format(i))
    axs[1, i].imshow(dct_gop[i], cmap='gray')
    axs[1, i].set_title('DCT Frame {}'.format(i))
plt.show()