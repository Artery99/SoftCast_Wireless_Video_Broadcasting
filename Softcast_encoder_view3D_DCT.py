# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:35:29 2023

@author: Hadi
"""

import cv2
import numpy as np

# Set the path to the 3D DCT YUV file
file_path = r'C:\akiyo_cif_dct.yuv'

# Set the video resolution and frame rate
width = 352
height = 288
fps = 30

# Open the 3D DCT YUV file
yuv_file = open(file_path, 'rb')

# Calculate the size of a single frame in bytes
frame_size = int(width * height * 3)

# Loop through all frames in the YUV file
for i in range(100): # loop over first 100 frames
    # Read the next frame from the 3D DCT YUV file
    frame = yuv_file.read(frame_size)
    
    # Convert the YUV frame to RGB
    yuv = np.frombuffer(frame, dtype=np.uint8)
    yuv = yuv.reshape((height, width, 3))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Display the result of the first frame
    if i == 0:
        cv2.imshow('3D DCT Frame', bgr)

    # Wait for the user to close the window
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Close the YUV file and destroy the window
yuv_file.close()
cv2.destroyAllWindows()
