import cv2
import numpy as np

# Load the two images
image1 = cv2.imread("to_send.jpg")
image2 = cv2.imread("stego.jpg")

# Convert images to RGB
to_hide_og = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
to_send_og = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Ensure the images have the same dimensions
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Convert images to arrays and normalize
array1 = np.array(image1)/255
array2 = np.array(image2)/255

# Calculate the absolute difference for each channel
abs_diff_r = np.abs(array1[:, :, 0] - array2[:, :, 0])
abs_diff_g = np.abs(array1[:, :, 1] - array2[:, :, 1])
abs_diff_b = np.abs(array1[:, :, 2] - array2[:, :, 2])

# Compute the mean absolute error (MAE) for each channel
mae_r = np.mean(abs_diff_r)
mae_g = np.mean(abs_diff_g)
mae_b = np.mean(abs_diff_b)

# Compute the average MAE across all channels
average_mae = (mae_r + mae_g + mae_b) / 3.0

# Calculate the percentage error
percentage_error = (average_mae) * 100.0

print("Mean Absolute Error (R):", mae_r)
print("Mean Absolute Error (G):", mae_g)
print("Mean Absolute Error (B):", mae_b)
print("Average Mean Absolute Error:", average_mae)
print(f"Percentage Error: {percentage_error}%")
