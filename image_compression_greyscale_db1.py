''' 
Image compression using the discrete wavelet transform
https://www.youtube.com/watch?v=eJLF9HeZA8I&t=18s

- Realiza a decomposição wavelet de uma imagem em escala de cinza em níveis múltiplos e exibe o resultado da decomposição como uma imagem

- Wavelet Decomposition of a Two-Dimensional array

- Mother wavelet: db1

biblioteca PyWavelets (pywt) -> realiza a decomposição 
biblioteca Matplotlib -> exibe a imagem resultante

decomposição wavelet -> técnica que separa os componentes de frequência de uma imagem, permitindo uma representação mais eficiente e compacta dos detalhes da imagem em diferentes escalas de resolução
'''

from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import pywt

plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

image = imread('lena.jpg')
grayscale = np.mean(image, -1); # Convert RGB to grayscale

## Wavelet decomposition (2 level)
n = 2
w = 'db1' ## Daubechies
coeffs = pywt.wavedec2(grayscale, wavelet=w, level=n)

# normalize each coefficient array
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level + 1]]

arr, coeff_slices = pywt.coeffs_to_array(coeffs)

plt.imshow(arr, cmap='gray', vmin=-0.25, vmax=0.75)
plt.show()
