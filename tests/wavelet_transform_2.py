'''
This version has noise in the output
'''

import numpy as np
import pywt
from PIL import Image

def embed_data(image_path, data):
    # Load the cover image
    cover_image = Image.open(image_path).convert('RGB')
    cover_array = np.array(cover_image)

    # Perform wavelet transformation on the cover image
    coeffs = pywt.dwt2(cover_array, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Convert the data into a binary string
    binary_data = ''.join(format(byte, '08b') for byte in data)

    # Add error correction bits (parity check)
    parity_bit = str(sum(int(bit) for bit in binary_data) % 2)  # Add a parity bit as the LSB of the entire message
    binary_data += parity_bit

    # Embed the encoded data into the wavelet coefficients
    cA_modified = np.copy(cA).astype(np.uint16)  # Convert to an appropriate data type

    for i in range(cA_modified.size):
        if binary_data[i] == '1':
            cA_modified.flat[i] |= 1  # Set the LSB to 1
        else:
            cA_modified.flat[i] &= ~1  # Set the LSB to 0

    # Reconstruct the modified coefficients
    modified_coeffs = (cA_modified, (cH, cV, cD))
    modified_image_array = pywt.idwt2(modified_coeffs, 'haar')

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

    # Extract the wavelet coefficients from the modified image
    coeffs = pywt.dwt2(modified_array, 'haar')
    cA_modified, (cH, cV, cD) = coeffs

    # Convert the wavelet coefficients to an appropriate data type for bitwise operations
    cA_modified = cA_modified.astype(np.uint16)

    # Flatten the coefficients array and convert to binary
    binary_data = np.unpackbits(cA_modified.view(np.uint8))

    # Perform error correction (parity check)
    data_without_parity = binary_data[:-1]  # Remove the last bit (parity bit)

    # Convert the binary data into bytes
    extracted_bytes = []
    for i in range(0, len(data_without_parity), 8):
        byte = int(''.join(map(str, data_without_parity[i:i + 8])), 2)
        extracted_bytes.append(byte)

    extracted_data = bytes(extracted_bytes).decode('latin-1')

    return extracted_data

# Embedding data into the cover image
image_path = 'image_test_low_resolution.png'
data_to_embed = b'\x01\x02\x03\x04\x05'  # Example binary data
embed_data(image_path, data_to_embed)

# Extracting data from the modified image
modified_image_path = 'modified_image.png'
extracted_data = extract_data(modified_image_path)
print('Extracted data:', extracted_data)
