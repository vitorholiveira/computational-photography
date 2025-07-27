import csv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show(img, title=''):
    a = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    plt.imshow(a)
    plt.axis('off')
    plt.title(title)
    plt.show()

def makehdr(images, exposure_times, gamma=1.0):
    merge_debvec = cv.createMergeDebevec()
    hdr = merge_debvec.process(images, times=exposure_times)

    tonemap = cv.createTonemap(gamma=gamma)
    ldr = tonemap.process(hdr)

    ldr_8bit = np.clip(ldr * 255, 0, 255).astype(np.uint8)

    return ldr_8bit

# PARAMS
# images -> one-dimensional numpy array containing LDR images
# exposure_time -> one-dimensional numpy array containing exposure times of the LDR images
# curve -> one-dimensional numpy array containing the 255 values of the camera's response curve.
def get_irradiance(images, exposure_time, curve):
    num_img = len(images)
    width, height, _ = images[0].shape
    irradiance = np.zeros((width, height, 3), dtype=np.float32)

    for row in range(height):
        for col in range(width):
            # Always reset the count of valid channel values to 6
            count = [6, 6, 6]

            # X[k][i,j] = exp(C[i,j])
            # E[i,j] = X[k][i,j]/T[k]

            for img_index in range(num_img):
                r = images[img_index][col,row][0]
                g = images[img_index][col,row][1]
                b = images[img_index][col,row][2]

                # Ignore channel value if it is 0 or 255
                # X[k][i,j] = exp(C[i,j])
                if r <= 0 or r >= 255:
                    count[0] -= 1
                else:
                    irradiance[col,row][0] += np.exp(curve[r][0]) / exposure_time[img_index]
                
                if g <= 0 or g >= 255:
                    count[1] -= 1
                else:
                    irradiance[col,row][1] += np.exp(curve[g][1]) / exposure_time[img_index]
                
                if b <= 0 or b >= 255:
                    count[2] -= 1
                else:
                    irradiance[col,row][2] += np.exp(curve[b][2]) / exposure_time[img_index]
            
            # Prevent division by 0
            if not count[0]:
                count[0] = 1
            if not count[1]:
                count[1] = 1
            if not count[2]:
                count[2] = 1

            # Calculate the average irradiance
            # E[i,j] = X[k][i,j]/T[k]
            irradiance[col,row][0] = irradiance[col,row][0] / count[0]
            irradiance[col,row][1] = irradiance[col,row][1] / count[1]
            irradiance[col,row][2] = irradiance[col,row][2] / count[2]

    return irradiance.astype(np.float32)

# PARAMS
# images -> one-dimensional numpy array containing LDR images (RGB)
# exposure_time -> one-dimensional numpy array containing exposure times of the LDR images (RGB)
# curve -> one-dimensional numpy array containing the 255 values of the camera's response curve.
def get_irradiance_opt(images, exposure_times, curve):
    irradiance_sum = np.zeros(images[0].shape, dtype=np.float32)
    exposure_times_reshaped = exposure_times[:, np.newaxis, np.newaxis]

    # Define valid pixels
    mask_valid_pixels = (images > 0) & (images < 255)
    num_valid_pixels = np.sum(mask_valid_pixels, axis=0)
    num_valid_pixels[num_valid_pixels == 0] = 1 # Prevent division by zero

    # X[k] = exp(C)
    # E = X[k]/T[k]    

    # Separate Channels
    r = images[...,0]
    g = images[...,1]
    b = images[...,2]
    curve_r = curve[:,0]
    curve_g = curve[:,1]
    curve_b = curve[:,2]

    # Calculate exposure using the camera’s response curve
    # X[k] = exp(C)
    exposure_r = np.exp(curve_r[r]) * mask_valid_pixels[...,0]
    exposure_g = np.exp(curve_g[g]) * mask_valid_pixels[...,1]
    exposure_b = np.exp(curve_b[b]) * mask_valid_pixels[...,2]

    # Divide the exposure by the time to obtain irradiance
    # Then sum the irradiance of all images to obtain the irradiance_sum
    # E = X[k]/T[k]
    irradiance_sum[...,0] = np.sum(exposure_r / exposure_times_reshaped, axis=0)
    irradiance_sum[...,1] = np.sum(exposure_g / exposure_times_reshaped, axis=0)
    irradiance_sum[...,2] = np.sum(exposure_b / exposure_times_reshaped, axis=0)

    # Calculate the average irradiance
    irradiance = irradiance_sum / num_valid_pixels

    return irradiance.astype(np.float32)

# PARAMS
# path -> string representing the file name concatenated with the path.
def read_camera_response_curve(path):
    curve = []
    with open(path, 'r') as src:
        reader = csv.reader(src, delimiter=' ')
        for row in reader:
            r = float(row[0])
            g = float(row[1])
            b = float(row[2])
            curve.append([r,g,b])
    return np.array(curve, dtype=np.float32)

# --------------------------------------------------------------------
# TONE MAPPING

# https://dl.acm.org/doi/pdf/10.1145/566654.566575

def tonemap_reinhard(img, gamma=1.0):
    img_processed = (img/255.0)**(1/gamma)
    luminance = (0.299*img_processed[...,0] + 0.587*img_processed[...,1] + 0.114*img_processed[...,2])

    n = luminance.size
    delta = 0.00001
    alpha = 0.18

    # L~ = exp((1/N) * sum(log(L[x,y]) + delta))
    avg_log_luminance = np.exp((1/n) * np.sum(np.log(luminance + delta)))

    # Ls[x,y] = (alpha/L~) * L[x,y]
    scaled_luminance = (alpha/avg_log_luminance) * luminance

    # Lg[x,y] = Ls[x,y] * (1 + Ls[x,y] / Lw[x,y]^2) / (1 + Ls[x,y])
    #global_luminance = scaled_luminance * (1 + scaled_luminance / scaled_luminance.max()**2) / (1 + scaled_luminance)

    # Lg[x,y] = Ls[x,y] / (1 + Ls[x,y])
    global_luminance = scaled_luminance  / (1 + scaled_luminance)

    global_luminance_ratio = global_luminance / luminance

    img_processed[:,:,0] = img_processed[:,:,0] * global_luminance_ratio
    img_processed[:,:,1] = img_processed[:,:,1] * global_luminance_ratio
    img_processed[:,:,2] = img_processed[:,:,2] * global_luminance_ratio

    return np.clip(img_processed*255,0,255).astype(np.uint8)

def tonemap_reinhard_local(img, gamma=1.0, threshold=0.03):
    img_processed = (img/255.0)**(1/gamma)
    luminance = (0.299*img_processed[...,0] + 0.587*img_processed[...,1] + 0.114*img_processed[...,2])

    n = luminance.size
    delta = 0.00001
    alpha = 0.18

    # L~ = exp((1/N) * sum(log(L[x,y]) + delta))
    avg_log_luminance = np.exp((1/n) * np.sum(np.log(luminance + delta)))

    # Ls[x,y] = (alpha/L~) * L[x,y]
    scaled_luminance = (alpha/avg_log_luminance) * luminance
    
    k_size = 1

    # | W[x,y,s] | < threshold
    # W[x,y,s(i)] = (V[x,y,s(i)] - V[x,y,s(i+1)]) / ((2^phi)*alpha*(s(i)^2) + V[x,y,s(i)])
    loop = True
    while loop:
        k_size += 2
        weight = center_surround(scaled_luminance, k_size, key=alpha)
        print(f'abs(max(weight)) = {np.max(np.abs(weight))}\nthreshold: {threshold}')
        loop = np.any(np.abs(weight) > threshold)

    # V[x,y,s(max)] = Ls[x,y] (x) W[x,y,s(max)]
    gaussian = cv.GaussianBlur(scaled_luminance, (k_size, k_size), 0)

    # Lg[x,y] = Ls[x,y] / (1 + V[x,y,s(max)])
    global_luminance = scaled_luminance / (1 + gaussian)

    global_luminance_ratio = global_luminance / luminance

    img_processed[:,:,0] = img_processed[:,:,0] * global_luminance_ratio
    img_processed[:,:,1] = img_processed[:,:,1] * global_luminance_ratio
    img_processed[:,:,2] = img_processed[:,:,2] * global_luminance_ratio
    
    return np.clip(img_processed*255,0,255).astype(np.uint8)

def center_surround(img, k_size, sharp=8.0, key=0.18):
    gaussian_1 = cv.GaussianBlur(img, (k_size, k_size), 0)
    gaussian_2 = cv.GaussianBlur(img, (k_size + 2, k_size + 2), 0)
    normalizer = ((2**sharp) * key / (k_size**2) + gaussian_1)

    # W[x,y,s(i)] = (V[x,y,s(i)] - V[x,y,s(i+1)]) / ((2^phi)*alpha*(s(i)^2) + V[x,y,s(i)])
    center = (gaussian_1 - gaussian_2) / normalizer

    return center

# --------------------------------------------------------------------

# https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz

def convert_sRGB_to_XZY(img):
    h, w, c = img.shape

    # Matrix that converts sRGB to XYZ
    matrix = np.array([[ 0.4124564, 0.3575761, 0.1804375 ],
                       [ 0.2126729, 0.7151522, 0.0721750 ],
                       [ 0.0193339, 0.1191920, 0.9503041 ]]).astype(np.float64)
    
    # sRGB values are rows
    srgb = (img.reshape(h*w,c) / 255).astype(np.float64)
    # sRGB values are now columns
    srgb = srgb.T
    # XYZ values are columns
    xyz = np.dot(matrix, srgb)
    # XYZ values are now rows
    xyz = xyz.T

    # Reconstruct the image
    img_xyz = xyz.reshape(h, w, c)
    img_xyz = np.clip(img_xyz, 0.0, 1.0).astype(np.float64)

    return img_xyz

def convert_XYZ_to_sRGB(img):
    h, w, c = img.shape

    # Matrix that converts XYZ to RGB
    matrix = np.array([[ 3.2404542,  -1.5371385,  -0.4985314 ],
                       [-0.9692660,   1.8760108,   0.0415560 ],
                       [ 0.0556434,  -0.2040259,   1.0572252 ]]).astype(np.float64)

    # XYZ values are rows
    xyz = img.reshape(h*w,c)
    # XYZ values are now columns
    xyz = xyz.T
    # sRGB values are columns
    srgb = np.dot(matrix, xyz)
    # sRGB values are now rows
    srgb = srgb.T

    # Reconstruct the image
    img_srgb = np.clip(np.round(srgb.reshape(h, w, c) * 255), 0, 255).astype(np.uint8)

    return img_srgb

# --------------------------------------------------------------------

# PARAMS
# img -> LDR image (RGB)
# pos_x -> horizontal index of the white pixel
# pos_y -> vertical index of the white pixel
def white_balance(img, pos_x, pos_y):
    val_r = img[pos_y, pos_x, 0]
    val_g = img[pos_y, pos_x, 1]
    val_b = img[pos_y, pos_x, 2]

    r = np.clip(img[..., 0]*(255/val_r), 0, 255).astype(np.uint8)
    g = np.clip(img[..., 1]*(255/val_g), 0, 255).astype(np.uint8)
    b = np.clip(img[..., 2]*(255/val_b), 0, 255).astype(np.uint8)

    img_wb = np.round(cv.merge((r, g, b))).astype(np.uint8)

    return img_wb