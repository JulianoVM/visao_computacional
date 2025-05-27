import numpy as np
import cv2
import os

# função para pegar número de imagens:
# def get_num_images(images_dir):
#     # num = 0

#     # for i in os.listdir(images_dir):
#     #     num = num + 1

#     return len(os.listdir(images_dir))

# função para pegar imagens:
def get_images(images_dir):
    caminho = []

    for nome in os.listdir(images_dir):
        caminho.append(os.path.join(images_dir, nome))

    return caminho

# função para carregar as imagens que pegamos:
def load_images(images_caminhos):
    images = []

    for images_caminho in images_caminhos:
        img = cv2.imread(images_caminho, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    return images

caminho_imagens = '.\imagens'

num_images = len(os.listdir(caminho_imagens))
images_caminhos = get_images(caminho_imagens)
images_carregadas = load_images(images_caminhos)
# images_redimensionadas = redimensionar(images_carregadas)

# imagens originais cinzas
for i in range(num_images):
    cv2.imshow('original cinza', images_carregadas[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# redimensionando imagens:
images_red = []

for i in range(num_images):
    red = cv2.resize(images_carregadas[i], (128,128))
    images_red.append(red)

for i in range(num_images):
    cv2.imshow('redimensionadas', images_red[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# aplicando o filtro gaussiano:
images_blur = []

for i in range(num_images):
    blur = cv2.GaussianBlur(images_carregadas[i], (5, 5), 0)
    images_blur.append(blur)

for i in range(num_images):
    cv2.imshow('filtro gaussiano', images_blur[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# aplicando a equalização de histograma:
images_equal = []

for i in range(num_images):
    equal = cv2.equalizeHist(images_carregadas[i])
    images_equal.append(equal)

for i in range(num_images):
    cv2.imshow('equalizacao de histograma', images_equal[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
