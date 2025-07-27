import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem original
original_image = cv2.imread('cat.jpg')
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Definir os níveis de compressão
compression_levels = [85, 50, 15]

# Listas para armazenar as imagens compostas e de diferença
composite_images = []
difference_images = []

for level in compression_levels:
    # Comprimir a imagem e salvar
    compressed_filename = f'compressed_{level}.jpg'
    cv2.imwrite(compressed_filename, original_image, [cv2.IMWRITE_JPEG_QUALITY, level])

    # Carregar a imagem comprimida
    compressed_image = cv2.imread(compressed_filename)
    compressed_image_rgb = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

    # Criar a imagem composta
    height, width, _ = original_image_rgb.shape
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    composite[:, :width//2] = original_image_rgb[:, :width//2]
    composite[:, width//2:] = compressed_image_rgb[:, width//2:]
    composite_images.append(composite)

    # Salvar a imagem composta
    composite_filename = f'composite_{level}.jpg'
    cv2.imwrite(composite_filename, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    # Calcular a diferença e escalá-la
    difference = np.abs(original_image_rgb.astype(np.float32) - composite.astype(np.float32))
    difference_scaled = np.clip(difference * 100, 0, 255).astype(np.uint8)
    difference_images.append(difference_scaled)

    # Salvar a imagem de diferença
    difference_filename = f'difference_{level}.jpg'
    cv2.imwrite(difference_filename, cv2.cvtColor(difference_scaled, cv2.COLOR_RGB2BGR))

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for i, (comp_img, diff_img, level) in enumerate(zip(composite_images, difference_images, compression_levels)):
    axes[i, 0].imshow(original_image_rgb)
    axes[i, 0].set_title(f'Original Image')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(comp_img)
    axes[i, 1].set_title(f'Composite Image (Quality {level})')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(diff_img)
    axes[i, 2].set_title(f'Difference Image (Quality {level})')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()
