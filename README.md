# SoftCast Wireless Video Broadcasting
This repository contains Python implementations of SoftCast encoding and decoding using Discrete Cosine Transform (DCT) and Hadamard Transform for efficient video transmission.

#Overview
SoftCast is a wireless video transmission scheme that eliminates quantization and entropy coding, instead using linear transforms to encode video for efficient transmission. This project applies DCT-based encoding, Hadamard Transform, and inverse transforms to process YUV video frames.

#Files and Their Functions

1️⃣ Softcast_encoder_3D_DCT.py
Reads a YUV video file.
Extracts luminance (Y) frames.
Forms a Group of Pictures (GoP).
Applies 3D DCT (spatial and temporal compression).
Displays both the original and DCT-transformed frames.

2️⃣ Softcast_encoder_view3D_DCT.py
Reads a 3D DCT-processed YUV file.
Converts frames from YUV to BGR (RGB-equivalent).
Displays the 3D DCT-transformed video frames.

3️⃣ Softcast_encoder_original_dct.py
Reads a YUV video and extracts Y (luminance) frames.
Applies 2D DCT to transform each frame into the frequency domain.
Displays the original frame alongside its DCT-transformed version.

4️⃣ Softcast_encoder_decoder.py
Performs SoftCast encoding and decoding:
Applies 3D DCT to the video.
Divides frames into chunks and computes variance-based scaling factors.
Uses the Hadamard Transform for signal spreading.
Applies inverse Hadamard Transform and inverse DCT (IDCT) to reconstruct the frames.
Displays the reconstructed video.
