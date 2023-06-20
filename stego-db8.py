import numpy as np
import pywt
from matplotlib import pyplot as plt
import cv2

def read_and_convert_image(image_path):
    # 1) reading files
    image = cv2.imread(image_path)
    # 2) converting to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image_subplot(image, subplot_number, title):
    plt.subplot(plot_rows, plot_columns, subplot_number)
    plt.axis('off')
    plt.title(title)
    plt.imshow(image, aspect="equal")

def dwt_coefficients(image, encoding_wavelet):
    cA, (cH, cV, cD) = pywt.dwt2(image, encoding_wavelet)
    return cA, cH, cV, cD

def show_coefficients_subplot(coefficients, subplot_base, dwt_labels):
    for i, a in enumerate(coefficients):
        subplot_number = subplot_base + i
        plt.subplot(plot_rows, plot_columns, subplot_number)
        plt.title(dwt_labels[i])
        plt.axis('off')
        plt.imshow(a, interpolation="nearest", cmap=plt.cm.gray)

def svd_decomposition(matrix):
    P, D, Q = np.linalg.svd(matrix, full_matrices=False)
    return P, D, Q

def save_image_to_file(image, filepath, figsize=None):
    fig = plt.figure(frameon=False)
    if figsize is not None:
        fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect="auto")
    fig.savefig(filepath)

encoding_wavelet = "db8"
decoding_wavelet = "db8"

plot_rows = 4
plot_columns = 4

to_hide_og = read_and_convert_image("to_hide.jpg")
to_send_og = read_and_convert_image("to_send.jpg")

# plot cover and image that will be hidden
show_image_subplot(to_send_og, 2, "Cover Image")
show_image_subplot(to_hide_og, 3, "Image to hide")

# --------------------------------------------
# Encoding Process
#---------------------------------------------

dimh, dimw, dimch = to_send_og.shape

# 3) seperating channels (colors) for cover and hidden images
to_send_r = to_send_og[:, :, 0]
to_send_g = to_send_og[:, :, 1]
to_send_b = to_send_og[:, :, 2]

to_hide_r = to_hide_og[:, :, 0]
to_hide_g = to_hide_og[:, :, 1]
to_hide_b = to_hide_og[:, :, 2]

# 4) taking wavelet transform for cover and hidden image
send_r, cHr, cVr, cDr = dwt_coefficients(to_send_r, encoding_wavelet)
send_g, cHg, cVg, cDg = dwt_coefficients(to_send_g, encoding_wavelet)
send_b, cHb, cVb, cDb = dwt_coefficients(to_send_b, encoding_wavelet)

hide_r, cHr1, cVr1, cDr1 = dwt_coefficients(to_hide_r, encoding_wavelet)
hide_g, cHg1, cVg1, cDg1 = dwt_coefficients(to_hide_g, encoding_wavelet)
hide_b, cHb1, cVb1, cDb1 = dwt_coefficients(to_hide_b, encoding_wavelet)

# plot all layers resulted from DWT
dwt_labels = ["Approximation", "Horizontal Detail", "Vertical Detail", "Diagonal Detail"]

show_coefficients_subplot([cAr, cHr, cVr, cDr], 5, dwt_labels)
show_coefficients_subplot([cAr1, cHr1, cVr1, cDr1], 9, dwt_labels)

print(cAr.shape)

# 5) compute svd for cover, hiding image

Pr, Dr, Qr = svd_decomposition(cAr)
Pg, Dg, Qg = svd_decomposition(cAg)
Pb, Db, Qb = svd_decomposition(cAb)

P1r, D1r, Q1r = svd_decomposition(cAr1)
P1g, D1g, Q1g = svd_decomposition(cAg1)
P1b, D1b, Q1b = svd_decomposition(cAb1)

print(Pr.shape, Dr.shape, Qr.shape)  # just for debugging

# 6) watermarking R,G,B channels using approximate values IE cA
# add hidden and cover
# merge original and cover parameters (svd)

S_wimgr = Dr + (0.10 * D1r)
S_wimgg = Dg + (0.10 * D1g)
S_wimgb = Db + (0.10 * D1b)

# 7) merging - get new values (R,G,B) using Pr S_wimgr and Qr --> basically reconstruct the SVD
# svd parameters into matrix coefficients

wimgr = np.dot(Pr * S_wimgr, Qr)

wimgg = np.dot(Pg * S_wimgg, Qg)

wimgb = np.dot(Pb * S_wimgb, Qb)

# cast type from merged r, g, b 
a = wimgr.astype(int) # red matrix
b = wimgg.astype(int) # green matrix
c = wimgb.astype(int) # blue matrix

# 8) merge reconstructed svd, this is using approximate values hence dimension/2
wimg = cv2.merge((a, b, c))
h, w, ch = wimg.shape

# 9) rgb coeffs for idwt, so that you can recreate a original img but with cA now having hidden info
# (cHr, cVr, cDr) -> wavelet coefficients corresponding to the horizontal (cHr), vertical (cVr) and diagonal (cDr) components obtained from the wavelet transform of the red channel
proc_r = wimg[:, :, 0], (cHr, cVr, cDr) # extracts pixel values only from the red channel of the image
proc_g = wimg[:, :, 1], (cHg, cVg, cDg)
proc_b = wimg[:, :, 2], (cHb, cVb, cDb)

# 3 stego images
# 10) performs the inverse of the discrete wavelet transform in each color channel (red, green and blue) of the processed image
# wavelet encoding -> parameter that indicates which waveform (wavelet) was used for the initial wavelet transform
processed_rgbr = pywt.idwt2(proc_r, encoding_wavelet)
processed_rgbg = pywt.idwt2(proc_g, encoding_wavelet)
processed_rgbb = pywt.idwt2(proc_b, encoding_wavelet)
# reconstructed color channels were obtained, in which the modified wavelet coefficients were incorporated back
# they represent the steganographed image, in which the hidden information is embedded in the modified wavelet coefficients

# combine color channels into a single image
wimghd = cv2.merge(
    (processed_rgbr.astype(int), processed_rgbg.astype(int), processed_rgbb.astype(int))
)

h, w, ch = wimghd.shape

# plot stego image
plt.subplot(plot_rows, plot_columns, 14)
plt.axis('off')
plt.title("Stego Image")
plt.imshow(wimghd, aspect="equal")

# --------------------------------------------
# Decoding Process
#---------------------------------------------

# 11) applying dwt to 3 stego channel images to get coeffs of stego image in R,G,B

Psend_r = pywt.dwt2(processed_rgbr, decoding_wavelet)
PcAr, (PcHr, PcVr, PcDr) = Psend_r

Psend_g = pywt.dwt2(processed_rgbg, decoding_wavelet)
PcAg, (PcHg, PcVg, PcDg) = Psend_g

Psend_b = pywt.dwt2(processed_rgbb, decoding_wavelet)
PcAb, (PcHb, PcVb, PcDb) = Psend_b

# 12) again do svd to decompose the approximate value PcAr
PPr, PDr, PQr = np.linalg.svd(PcAr, full_matrices=False)
PPg, PDg, PQg = np.linalg.svd(PcAg, full_matrices=False)
PPb, PDb, PQb = np.linalg.svd(PcAb, full_matrices=False)

# 13) subtract from R,G,B channels values of cover image
S_ewatr = (PDr - Dr) / 0.10
S_ewatg = (PDg - Dg) / 0.10
S_ewatb = (PDb - Db) / 0.10

# 14) merging -> merge new approximations with hidden SVD found earlier
ewatr = np.dot(P1r * S_ewatr, Q1r)
ewatg = np.dot(P1g * S_ewatg, Q1g)
ewatb = np.dot(P1b * S_ewatb, Q1b)

# 15) merge recreate hidden image - still  based on approximations, hence dim /2
d = ewatr.astype(int)
e = ewatg.astype(int)
f = ewatb.astype(int)
eimg = cv2.merge((d, e, f))

# 16) coeffs of original hidden image except the new derived appproximation
eproc_r = eimg[:, :, 0], (cHr1, cVr1, cDr1)
eproc_g = eimg[:, :, 1], (cHg1, cVg1, cDg1)
eproc_b = eimg[:, :, 2], (cHb1, cVb1, cDb1)

# 17) hidden stego images get high res r,g,b seperate images/channels usign idwt
eprocessed_rgbr = pywt.idwt2(eproc_r, decoding_wavelet)
eprocessed_rgbg = pywt.idwt2(eproc_g, decoding_wavelet)
eprocessed_rgbb = pywt.idwt2(eproc_b, decoding_wavelet)
# plt.figure(6)

# just converting float to int prior to cv2.merge
x1 = eprocessed_rgbr.astype(int)

y1 = eprocessed_rgbg.astype(int)

z1 = eprocessed_rgbb.astype(int)

# 18) combine different high res r,g,b to get hidden image 
# figure 9 is final output
hidden_rgb = cv2.merge((x1, y1, z1))

h1, w1, ch1 = hidden_rgb.shape

plt.subplot(plot_rows, plot_columns, 15)
plt.axis('off')
plt.title("Decoded Hidden Image")
plt.imshow(hidden_rgb, aspect="equal")

plt.show()
plt.close()

# save stego image to filesystem
save_image_to_file(wimghd, "stego.jpg", figsize=(float(w) / 100, float(h) / 100))

# save decoded hidden image to filesystem
save_image_to_file(hidden_rgb, "hidden_rgb.jpg", figsize=(7.20, 4.80))

