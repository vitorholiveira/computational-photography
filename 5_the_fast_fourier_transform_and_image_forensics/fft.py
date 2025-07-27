import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

def fft_1d(x):
    N = len(x)
    if N <= 1:
        return x
    
    # Chamada recursiva para calcular a FFT dos elementos de índice par
    even = fft_1d(x[0::2])
    
    # Chamada recursiva para calcular a FFT dos elementos de índice ímpar
    odd = fft_1d(x[1::2])
    
    # Calcula os fatores de rotação (Twiddle Factors) para combinar os resultados pares e ímpares
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    
    # Combina os resultados dos pares e ímpares para formar a FFT completa
    return np.concatenate([even + T, even - T])


def ifft_1d(X):
    N = len(X)
    if N <= 1:
        return X
    
    # Chamada recursiva para calcular a IFFT dos elementos de índice par
    even = ifft_1d(X[0::2])
    
    # Chamada recursiva para calcular a IFFT dos elementos de índice ímpar
    odd = ifft_1d(X[1::2])
    
    # Calcula os fatores de rotação (Twiddle Factors) para combinar os resultados pares e ímpares
    T = np.exp(2j * np.pi * np.arange(N // 2) / N) * odd
    
    # Combina os resultados dos pares e ímpares para formar a IFFT completa
    return np.concatenate([(even + T) / 2, (even - T) / 2])

def fft_2d(matrix):
    # Aplicar a FFT 1D para cada linha
    fft_rows = np.array([fft_1d(row) for row in matrix])
    # Aplicar a FFT 1D para cada coluna do resultado
    fft_cols = np.array([fft_1d(col) for col in fft_rows.T]).T
    return fft_cols

def ifft_2d(matrix):
    # Aplicar a IFFT 1D para cada linha
    ifft_rows = np.array([ifft_1d(row) for row in matrix])
    # Aplicar a IFFT 1D para cada coluna do resultado
    ifft_cols = np.array([ifft_1d(col) for col in ifft_rows.T]).T
    return ifft_cols

if __name__ == "__main__":
    image = np.array(cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE))

    start = time.time()
    fft_result = fft_2d(image)
    end = time.time()
    fft_time = end - start

    start = time.time()
    fft_result_np = np.fft.fft2(image)
    end = time.time()
    fft_np_time = end - start

    start = time.time()
    ifft_result = ifft_2d(fft_result)
    end = time.time()
    ifft_time = end - start

    start = time.time()
    ifft_result_np = np.fft.ifft2(fft_result_np)
    end = time.time()
    ifft_np_time = end - start

    print('fft time: ' + str(fft_time))
    print('fft np time: ' + str(fft_np_time))
    print('ifft time: ' + str(ifft_time))
    print('ifft np time: ' + str(ifft_np_time))

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('DFT Spectrum')
    plt.imshow(np.fft.fftshift(np.log(np.abs(fft_result) + 1)), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Recovered DFT -> IDFT')
    plt.imshow(np.abs(ifft_result), cmap='gray')
    plt.axis('off')

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Original - NumPy')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('DFT Spectrum - NumPy')
    plt.imshow(np.fft.fftshift(np.log(np.abs(fft_result_np) + 1)), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Recovered DFT -> IDFT - NumPy')
    plt.imshow(np.abs(fft_result_np), cmap='gray')
    plt.axis('off')

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('DFT Spectrum')
    plt.imshow(np.fft.fftshift(np.log(np.abs(fft_result) + 1)), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('DFT Spectrum - NumPy')
    plt.imshow(np.fft.fftshift(np.log(np.abs(fft_result_np) + 1)), cmap='gray')
    plt.axis('off')

    plt.show()
