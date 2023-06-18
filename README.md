### DSP - Steganography with Python

Processo de esconder a imagem dentro de outra imagem:

1) Ler os arquivos e convertê-los para o formato RGB
2) Separar os canais (cores) para a imagem de capa (cover) e a imagem oculta (hidden)
3) Aplicar a transformada wavelet na cover e hidden image
4) Realizar a decomposição em valores singulares (SVD) da cover e hidden image
5) Incorporar a informação oculta nos parâmetros 'D' da cover image
6) Reconstruir a matriz de coeficientes a partir dos parâmetros SVD incorporados
7) Concatenar os três canais RGB reconstruídos em uma única matriz
8) Extrair os coeficientes horizontais, verticais e diagonais de cada canal (RGB) da imagem
9) Aplicar a transformada inversa em cada canal da imagem processada gerando a imagem esteganografada (stego image)

Processo de reverter a estenografia:

10) Aplicar a transformada decodificadora em cada canal da imagem esteganografada (stego image)
11) Realizar a decomposição em valores singulares (SVD) da imagem esteganografada
12) Reverter as informações incluídas no parametro 'D' da cover image na etapa 5 através da operação inversa
13) Recombinar as aproximações com os valores SVD ocultos para reconstruir a imagem oculta
14) Obter a hidden image reconstruída, que consiste nos canais de cor combinados com as diferenças normalizadas do SVD 
15) Extrair os coeficientes horizontais, verticais e diagonais de cada canal (RGB) da hidden image
16) Aplicar a transformada inversa em cada canal da imagem para gerar a imagem final da informação oculta

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
