import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift

def dft(image):
    # Obtém as dimensões da imagem
    M, N = image.shape
    
    # Cria arrays representando os índices dos pixels
    x = np.arange(M)  # Índices das linhas
    y = np.arange(N)  # Índices das colunas
    
    # Redimensiona os índices para facilitar a multiplicação de matrizes
    u = x.reshape((M, 1))  # Índices das linhas redimensionados como vetor coluna
    v = y.reshape((N, 1))  # Índices das colunas redimensionados como vetor coluna
    
    # Calcula os exponenciais para a fórmula da DFT
    exp_xu = np.exp(-2j * np.pi * u * x / M)  # Exponencial para as linhas
    exp_yv = np.exp(-2j * np.pi * v * y / N)  # Exponencial para as colunas

    # Realiza a DFT 2D por multiplicação de matrizes
    dft_result = np.dot(exp_xu, np.dot(image, exp_yv))
    
    return dft_result

def idft(dft_coefficients):
    # Obtém as dimensões do array de coeficientes DFT
    M, N = dft_coefficients.shape
    
    # Cria arrays representando os índices de frequência
    x = np.arange(M)  # Índices de frequência para as linhas
    y = np.arange(N)  # Índices de frequência para as colunas
    
    # Redimensiona os índices para facilitar a multiplicação de matrizes
    u = x.reshape((M, 1))  # Índices de frequência para as linhas redimensionados como vetor coluna
    v = y.reshape((N, 1))  # Índices de frequência para as colunas redimensionados como vetor coluna
    
    # Calcula os exponenciais para a fórmula da IDFT
    exp_xu = np.exp(2j * np.pi * u * x / M)  # Exponencial para as linhas
    exp_yv = np.exp(2j * np.pi * v * y / N)  # Exponencial para as colunas
    
    # Realiza a IDFT 2D por multiplicação de matrizes
    idft_result = np.dot(exp_xu, np.dot(dft_coefficients, exp_yv))
    
    # Limita o resultado ao intervalo válido para dados de imagem e converte para uint8
    return np.clip(idft_result.real, 0, 255).astype(np.uint8)

def q1(image):
    dft_result = dft(image)
    dft_result_real = dft_result.real
    dft_result_imag = dft_result.imag
    
    idft_result = idft(dft_result)
    idft_result_real = idft(dft_result_real + 0j)
    idft_result_imag = idft(1j * dft_result_imag)
    
    cv2.imwrite('./result/idft_result.jpeg', idft_result)
    cv2.imwrite('./result/idft_result_real.jpeg', idft_result_real)
    cv2.imwrite('./result/idft_result_imag.jpeg', idft_result_imag)

    cv2.imshow('image', image)
    cv2.imshow('idft result', idft_result)
    cv2.imshow('idft result real', idft_result_real)
    cv2.imshow('idft result imag', idft_result_imag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def q2(image):
    a = idft(fft2(image))
    b = np.clip(ifft2(dft(image)).real, 0, 255).astype(np.uint8)
    c = idft(dft(image))

    cv2.imwrite('./result/2a.jpeg', a)
    cv2.imwrite('./result/2b.jpeg', b)
    cv2.imwrite('./result/2c.jpeg', c)

    cv2.imshow('2a', a)
    cv2.imshow('2b', b)
    cv2.imshow('2c', c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def q3(image):
    a = dft(image)
    a[0,0] = 0
    a = idft(a)
    b = np.clip(image - np.mean(image), 0, 255).astype(np.uint8)
    
    cv2.imwrite('./result/3a.jpeg', a)
    cv2.imwrite('./result/3b.jpeg', b)

    cv2.imshow('3a', a)
    cv2.imshow('3b', b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def q4(image):
    a = idft(fftshift(dft(image)))
    x, y = np.indices(image.shape)
    mask = (-1) ** (x + y)
    b = np.clip(image * mask, 0, 255).astype(np.uint8)
    c = fftshift(image)

    cv2.imwrite('./result/4a.jpeg', a)
    cv2.imwrite('./result/4b.jpeg', b)
    cv2.imwrite('./result/4c.jpeg', c)

    cv2.imshow('4a', a)
    cv2.imshow('4b', b)
    cv2.imshow('4c', c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def q5(image):
    frequences = fft2(image)
    amplitude_spectrum = np.abs(frequences)
    phase_spectrum = np.angle(frequences)
    amplitude_idft = np.clip(ifft2(amplitude_spectrum), 0, 255).astype(np.uint8)
    phase_idft = cv2.normalize(np.abs(ifft2(np.exp(1j * phase_spectrum))), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite('./result/5a.jpeg', amplitude_idft)
    cv2.imwrite('./result/5b.jpeg', phase_idft)

    cv2.imshow('5a', amplitude_idft)
    cv2.imshow('5b', phase_idft)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    image = np.array(cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE))

    #q1(image)

    #q2(image)

    #q3(image)

    q4(image)

    #q5(image)
