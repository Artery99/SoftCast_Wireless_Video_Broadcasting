# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:12:28 2023

@author: Hadi
"""

import cv2
import numpy as np
import yuvio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fftpack import dct, idct
from scipy.linalg import hadamard

# Set the path to the YUV file
file_path = r'akiyo_cif.y4m'

# Set the video resolution and frame rate
width = 352
height = 288
fps = 30

# Set the number of frames in the GoP and the chunk size
gop_size = 5
chunk_size = 16

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

# Extract chunks from each frame and put them in a matrix X then calculate its corresponding matrix and put it in Variances
X = np.empty((gop_size, height//chunk_size, width//chunk_size, chunk_size, chunk_size))
Variances = np.empty((gop_size, height//chunk_size, width//chunk_size))
for i in range(gop_size):
    for j in range(0, height, chunk_size):
        for k in range(0, width, chunk_size):
            chunk = dct_gop[i, j:j+chunk_size, k:k+chunk_size]
            X[i, j//chunk_size, k//chunk_size] = chunk
            Variances[i, j//chunk_size, k//chunk_size] = np.var(chunk)

# Calculate the scaling factor for each chunk
sum_sqrt_variances = np.sum(np.sqrt(Variances))
g_factors = np.empty((gop_size, height//chunk_size, width//chunk_size))
for i in range(gop_size):
    for j in range(0, height, chunk_size):
        for k in range(0, width, chunk_size):
            g = (Variances[i, j//chunk_size, k//chunk_size]**(-1/4)) * np.sqrt(100 / sum_sqrt_variances)
            g_factors[i, j//chunk_size, k//chunk_size] = g
            
# Multiply each chunk by its corresponding scaling factor
X_scaled = np.empty_like(X)
for i in range(gop_size):
    for j in range(0, height, chunk_size):
        for k in range(0, width, chunk_size):
            chunk = X[i, j//chunk_size, k//chunk_size] * g_factors[i, j//chunk_size, k//chunk_size]
            X_scaled[i, j//chunk_size, k//chunk_size] = chunk
            
X_scaled_2d = X_scaled.reshape((gop_size * height//chunk_size * width//chunk_size, chunk_size * chunk_size))
# This will reshape X_scaled into a 2D matrix where each row contains the flattened version of a chunk
# Each column contains the elements of a chunk in row-major order.

# size of the Hadamard matrix
n = 256

# Construct the Hadamard matrix
H = hadamard(n)

# Multiply X_scaled_2d by the Hadamard matrix
output = X_scaled_2d.dot(H)
# Hadamard-transformed version of X_scaled_2d.

# Decoder

# Considering High SNR implies small noise, the entries in (sigma matrix) approach 0.
# Hence: X(llse) = C^(-1) Y // C = H.G
# In other words we are dividing by scaling factors and doing the inverse hadamard transform.

# Perform inverse Hadamard transform
H_inv = -1 * H
X_scaled_2d_inv = output.dot(H_inv)

# Reshape X_scaled_2d_inv to match the original shape of X_scaled
X_inv = X_scaled_2d_inv.reshape((gop_size, height//chunk_size, width//chunk_size, chunk_size, chunk_size))

# Divide each chunk by its corresponding scaling factor
X_dct = np.empty_like(dct_gop)
for i in range(gop_size):
    for j in range(0, height, chunk_size):
        for k in range(0, width, chunk_size):
            chunk = X_inv[i, j//chunk_size, k//chunk_size] / g_factors[i, j//chunk_size, k//chunk_size]
            X_dct[i, j:j+chunk_size, k:k+chunk_size] = chunk

# Perfrom IDCT to reconstruct the GoP
reconstructed_gop = idct(idct(idct(X_dct, axis=2, norm='ortho'), axis=1, norm='ortho'), axis=0, norm='ortho')

# Show the first frame of the reconstructed GoP
imgplot = plt.imshow(reconstructed_gop[0], cmap = "gray")
plt.show()





