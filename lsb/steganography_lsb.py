'''
Least significant bit (LSB) modification technique to hide messages inside pictures
the binary representation of each character in the message is embedded into the least significant bit of the pixel values in the grayscale image

This version has some noise introduced during the process of hiding the message 
'''

from PIL import Image
import numpy as np

def hide_message_in_image(image_path, message, output_path):
    image = Image.open(image_path)
    grayscale = np.array(image.convert('L'))  # Convert image to grayscale

    # Flatten the grayscale image
    flattened = grayscale.flatten()

    # Convert the message to binary
    binary_message = ''.join(format(ord(c), '08b') for c in message)

    # Check if the message can fit in the image
    if len(binary_message) > len(flattened):
        raise ValueError("Message is too long to be hidden in the image.")

    # Modify the least significant bit of each pixel
    for i in range(len(binary_message)):
        bit = int(binary_message[i])
        flattened[i] = (flattened[i] & 0xFE) | bit

    # Reshape the modified flattened image
    modified_image = flattened.reshape(grayscale.shape)

    # Convert grayscale image to RGB for saving
    modified_image = Image.fromarray(np.uint8(modified_image), mode='L')

    # Save the modified grayscale image
    modified_image.save(output_path)

    print(f"Message hidden in image: {message}")
    print(f"Modified image saved to: {output_path}")


def decode_message_from_image(image_file):
    image = Image.open(image_file)
    grayscale = np.array(image.convert('L'))  # Convert image to grayscale

    # Flatten the grayscale image
    flattened = grayscale.flatten()

    # Extract the least significant bit of each pixel
    binary_message = ''.join(str(pixel & 1) for pixel in flattened)

    # Convert binary message to text
    decoded_message = ''
    for i in range(0, len(binary_message), 8):
        byte = binary_message[i:i + 8]
        if byte == '00000000':
            break
        decoded_byte = int(byte, 2)
        decoded_message += chr(decoded_byte)

    return decoded_message


image_path = 'lena.jpg'
message = "Hello, world!"
output_path = 'stego_image.png'

hide_message_in_image(image_path, message, output_path)

decoded_message = decode_message_from_image(output_path)
print("Decoded message:", decoded_message)
