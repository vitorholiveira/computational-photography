import numpy as np
import rawpy as rp
import cv2


def main():
    # Open the DNG file
    dng_path = './img/scene_raw.dng'
    raw = rp.imread(dng_path)
    cfa = raw.raw_image_visible
    
    # Demosaicking
    img, cfa_rgb = demosaic_bayer(cfa)
    cv2.imwrite('./result/img.jpeg', img)
    cv2.imwrite('./result/cfa_rgb.jpeg', cfa_rgb)

    # White Balance
    pos_x_paper = 2269
    pos_y_paper = 2342
    img_wb_paper = white_balance(img, pos_x_paper, pos_y_paper)
    cv2.imwrite('./result/white_balance_paper.jpeg', img_wb_paper)
    pos_x_icon = 1704
    pos_y_icon = 752
    img_wb_icon = white_balance(img, pos_x_icon, pos_y_icon)
    cv2.imwrite('./result/white_balance_icon.jpeg', img_wb_icon)

    # Gamma Encoding
    gamma = 2.2
    img_gamma_encoding_paper = gamma_correction(img_wb_paper, 1/gamma)
    cv2.imwrite('./result/gamma_encoding_paper.jpeg', img_gamma_encoding_paper)
    img_gamma_encoding_icon = gamma_correction(img_wb_icon, 1/gamma)
    cv2.imwrite('./result/gamma_encoding_icon.jpeg', img_gamma_encoding_icon)

    # Gamma Correction
    gamma = 1.5
    img_gamma_correction_paper = gamma_correction(img_wb_paper, gamma)
    cv2.imwrite('./result/gamma_correction_paper.jpeg', img_gamma_correction_paper)
    img_gamma_correction_icon = gamma_correction(img_wb_icon, gamma)
    cv2.imwrite('./result/gamma_correction_icon.jpeg', img_gamma_correction_icon)

'''
    1) Demosaicking
'''
# Demosaic the Bayer CFA image [ R G G B ] 12 bits -> 8 bits
def demosaic_bayer(cfa):
    # Define Bayer filter masks
    red_mask = np.zeros_like(cfa, dtype=bool)
    red_mask[1::2, 1::2] = 1

    blue_mask = np.zeros_like(cfa, dtype=bool)
    blue_mask[::2, ::2] = 1

    green_mask = ~(red_mask | blue_mask)

    # CFA is 12 bits, convert to 8 bits (2^4 = 16)
    new_cfa = cfa / 16

    # Assign values to RGB channels
    r = new_cfa * red_mask
    g = new_cfa * green_mask
    b = new_cfa * blue_mask

    cfa_rgb = cv2.merge((r, g, b))
    cfa_rgb = np.clip(cfa_rgb, 0, 255).astype(np.uint8)

    img = bilinear_interpolation(r, g, b)

    return img, cfa_rgb

# Bilinear interpolation [ R G G B ]
def bilinear_interpolation(red, green, blue):
    r = red
    g = green
    b = blue
    # Red pixels
    r += (np.roll(r, 1, axis = 0) + np.roll(r, -1, axis = 0))/2
    r += (np.roll(r, 1, axis = 1) + np.roll(r, -1, axis = 1))/2
    # Green pixels
    g += (np.roll(g, 1, axis = 0) + np.roll(g, -1, axis = 0) + np.roll(g, 1, axis = 1) + np.roll(g, -1, axis = 1))/4
    # Blue pixels
    b += (np.roll(b, 1, axis = 0) + np.roll(b, -1, axis = 0))/2
    b += (np.roll(b, 1, axis = 1) + np.roll(b, -1, axis = 1))/2

    rgb = cv2.merge((r, g, b))
    return np.clip(rgb, 0, 255).astype(np.uint8)

'''
    2) White Balance
'''
def white_balance(img, pos_x, pos_y):
    val_r = img[pos_y, pos_x, 0]
    val_g = img[pos_y, pos_x, 1]
    val_b = img[pos_y, pos_x, 2]

    r = np.clip(img[..., 0]*(255/val_r)).astype(np.uint8)
    g = np.clip(img[..., 1]*(255/val_g)).astype(np.uint8)
    b = np.clip(img[..., 2]*(255/val_b)).astype(np.uint8)

    img_wb = cv2.merge((r, g, b))

    return img_wb

'''
    3) Gamma Encoding/Correction
'''
def gamma_correction(img, gamma):

    img_normalized = img / 255.0

    img_gamma = np.power(img_normalized, gamma) * 255
    img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)
    
    return img_gamma

if __name__ == "__main__":
    main()