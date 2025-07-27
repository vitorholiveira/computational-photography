import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def build_gaussian_sequence(image, levels, ksize=5):
    gaussian_sequence = [image] 
    for i in range(levels):
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        gaussian_sequence.append(image)
    return gaussian_sequence

def build_laplacian_sequence(gaussian_sequence):
    laplacian_sequence = []
    for i in range(len(gaussian_sequence) - 1):
        laplacian = gaussian_sequence[i] - gaussian_sequence[i + 1]
        laplacian_sequence.append(laplacian)
    laplacian_sequence.append(gaussian_sequence[-1])
    return laplacian_sequence

def blend_images(laplacianA, laplacianB, gaussian_mask):
    blended_image = np.zeros(laplacianA[0].shape, dtype=np.float32)
    for la, lb, gm in zip(laplacianA, laplacianB, gaussian_mask):
        gm = cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR)
        blended_image += gm * la + (1 - gm) * lb
    return blended_image

def laplacian_blend(A, B, M, levels=10, ksize=5):
    # Generate Gaussian sequences for A, B, and M
    gaussianA = build_gaussian_sequence(A, levels, ksize)
    gaussianB = build_gaussian_sequence(B, levels, ksize)
    gaussianM = build_gaussian_sequence(M, levels, ksize)

    # Generate Laplacian sequences for A and B
    laplacianA = build_laplacian_sequence(gaussianA)
    laplacianB = build_laplacian_sequence(gaussianB)

    blended_image = blend_images(laplacianA, laplacianB, gaussianM)

    return blended_image

def part_two():
    A = cv2.imread('Golf.png').astype(np.float32)
    B = cv2.imread('Tennis.png').astype(np.float32)

    # Generate the binary mask
    rows, cols, _ = A.shape
    M = np.zeros((rows, cols), dtype=np.float32)
    M[rows // 2:,:] = 1

    levels = 25 

    ksize = 21

    blended_image = laplacian_blend(A, B, M, levels, ksize)

    plt.imshow(cv2.cvtColor(blended_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    cv2.imwrite(f'my_blended_image_{levels}_levels.jpg', blended_image)

def part_three():
    foreground = cv2.imread('fg_framed.jpeg').astype(np.float32)
    background = cv2.imread('eu.jpeg').astype(np.float32)
    mask = cv2.imread('mask_framed.jpeg').astype(np.float32)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255

    levels = 5

    ksize = 3

    blended_image = laplacian_blend(foreground, background, mask, levels, ksize)

    plt.imshow(cv2.cvtColor(blended_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    cv2.imwrite(f'result_{levels}_levels.jpg', blended_image)

part_three()
