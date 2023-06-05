import pywt
import numpy as np
import cv2

# Carregar a imagem
imagem = cv2.imread('lena.jpg', 0)

# Definir a wavelet a ser utilizada (Daubechies)
wavelet = 'db4'

# Realizar a DWT 2D na imagem
coeffs = pywt.wavedec2(imagem, wavelet)

# Definir o nível de decomposição desejado
nivel = 3

# Aplicar a compressão de coeficientes descartando detalhes de alta frequência
coeffs_nivel = list(coeffs)
for i in range(1, nivel + 1):
    coeffs_nivel[i] = tuple([np.zeros_like(v) for v in coeffs_nivel[i]])

# Realizar a reconstrução da imagem utilizando apenas os coeficientes preservados
imagem_reconstruida = pywt.waverec2(coeffs_nivel, wavelet)

# Converter a imagem reconstruída para o tipo uint8 (imagem em tons de cinza)
imagem_reconstruida = np.uint8(imagem_reconstruida)

# Exibir a imagem original e a imagem reconstruída
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem Reconstruída', imagem_reconstruida)
cv2.waitKey(0)
#cv2.destroyAllWindows()
