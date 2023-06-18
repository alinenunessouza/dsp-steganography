'''Realiza a incorporação e extração de dados em uma imagem usando a transformada wavelet'''

'''WARNING: em análise, código não está totalmente funcional'''

import numpy as np
import pywt
from PIL import Image
from matplotlib import pyplot as plt

def embed_data(image_path, data):
    image2 = Image.open(image_path)    
    # Load the cover image
    image = Image.open(image_path).convert('L')
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(image)
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(image2)
    plt.show()
    # image.show()
    
    image_as_array = np.array(image)
    print(image_as_array)

    # Perform wavelet transformation on the cover image
    coeffs = pywt.dwt2(image_as_array, 'db1')
    cA, (cH, cV, cD) = coeffs

    # cAImage = Image.fromarray(np.uint8(cA), mode='RGB')
    # cAImage.show()

    # Convert the data into a binary string
    binary_data = ''.join(format(ord(char), '08b') for char in data)
    print('binary_data', binary_data)

    # Determine the size difference between cA and the other coefficients
    size_diff = len(cH) - len(cA)

    # Resize cA to match the size of cH, cV, and cD
    if size_diff > 0:
        cA_resized = np.pad(cA, ((0, size_diff), (0, size_diff)), mode='constant')
    elif size_diff < 0:
        cA_resized = cA[:len(cH), :len(cH)]
    else:
        cA_resized = cA

    # Embed the data into the resized cA coefficients
    for i, bit in enumerate(binary_data):
        if bit == '1':
            cA_resized[i // 2, i % 2] += 1

    # Reconstruct the modified coefficients
    modified_coeffs = (cA_resized, (cH, cV, cD))
    modified_image_array = pywt.idwt2(modified_coeffs, 'db1')

    # Convert the modified image array to 8-bit integers
    modified_image_array = modified_image_array.clip(0, 255).astype(np.uint8)

    # Create the modified image
    modified_image = Image.fromarray(modified_image_array)

    # Save the modified image
    modified_image.save('modified_image.png')
    print('Data embedded successfully!')

def extract_data(image_path):
    # Load the modified image
    modified_image = Image.open(image_path).convert('RGB')
    modified_array = np.array(modified_image)

    # Perform wavelet transformation on the modified image
    coeffs = pywt.dwt2(modified_array, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Extract the embedded data from the wavelet coefficients
    extracted_data = ''
    for i in range(len(cA.flatten()) - 1):
        if np.any(cA.flatten()[i] % 2 == 1):  # Check if any element in cA.flatten()[i] is odd
            extracted_data += '1'
        else:
            extracted_data += '0'

    # Convert the binary string to characters
    extracted_chars = []
    for j in range(0, len(extracted_data), 8):
        char = chr(int(extracted_data[j:j + 8], 2))
        extracted_chars.append(char)

    extracted_text = ''.join(extracted_chars)

    return extracted_text

# Embedding data into the cover image
image_path = 'lena.jpg'
data_to_embed = 'Hello, this is a secret message!'
embed_data(image_path, data_to_embed)

# # Extracting data from the modified image
# modified_image_path = 'modified_image.png'
# extracted_data = extract_data(modified_image_path)
# print('Extracted data:', extracted_data)