### DSP - Steganography with Python

#### Process of hiding an image within another image:

1) Read the files (cover and hidden image).
2) Convert the files to the RGB format.
3) Separate the channels (colors) for the cover image and the hidden image.
4) Apply the wavelet transform to the cover and hidden images.
5) Perform Singular Value Decomposition (SVD) on the cover and hidden images.
6) Embed the hidden information into the 'D' parameters of the cover image.
7) Reconstruct the coefficient matrix from the embedded SVD parameters.
8) Concatenate the three reconstructed RGB channels into a single matrix.
9) Extract the horizontal, vertical, and diagonal coefficients from each RGB channel of the image.
10) Apply inverse transform to each channel of the processed image, generating the stego image.

#### Steganography reversal process::

11) Apply the decoding transform to each channel of the stego image.
12) Perform Singular Value Decomposition (SVD) on the stego image.
13) Reverse the information embedded in the 'D' parameter of the cover image in step 5 through the inverse operation.
14) Combine the approximations with the hidden SVD values to reconstruct the hidden image.
15) Obtain the reconstructed hidden image, which consists of the color channels combined with the normalized SVD differences.
16) Extract the horizontal, vertical, and diagonal coefficients from each RGB channel of the hidden image.
17) Apply inverse transform to each channel of the image to generate the final hidden information image.

#### Project Setup

1. Create a virtual python environment
````
python3 -m venv env
````
2. Activate virtual environment
    - Linux / macOS
    ````
    source env/bin/activate
    ````
    - Windows
    ```
    .\env\Scripts\activate
    ```
3. Install packages from `requirements.txt`
````
pip install -r requirements.txt
````

#### Add new packages to `requirements.txt`
1. Install the package into the local environment
```
pip install <PACKAGE>
```
2. Update the requirements.txt
```
pip freeze > requirements.txt
```
