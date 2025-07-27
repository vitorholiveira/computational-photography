import cv2
import numpy as np

# Global variables for mouse callback
drawing = False  # True if mouse is pressed
mode = 0  # 0: background, 1: foreground, 2: rectangle
ix, iy = -1, -1
rect = None

img = cv2.imread('dog2.jpg')
if img is None:
    print("Could not open or find the image")
    exit()

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
img_display = img.copy()

def draw_scribble(event, x, y, flags, param):
    global ix, iy, drawing, mode, mask, img, img_display, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        if mode == 2:
            rect = (ix, iy, x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode == 0:
                cv2.circle(img_display, (x, y), 3, (0, 0, 0), -1)
                cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
            elif mode == 1:
                cv2.circle(img_display, (x, y), 3, (255, 255, 255), -1)
                cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
            elif mode == 2:
                img_display = img.copy()
                cv2.rectangle(img_display, (ix, iy), (x, y), (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == 2:
            rect = (ix, iy, x, y)
            cv2.rectangle(img_display, (ix, iy), (x, y), (255, 0, 0), 2)
        elif mode == 0:
            cv2.circle(img_display, (x, y), 3, (0, 0, 0), -1)
            cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
        elif mode == 1:
            cv2.circle(img_display, (x, y), 3, (255, 255, 255), -1)
            cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_scribble)

while True:
    cv2.imshow('image', img_display)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('0'):
        mode = 0
        print("Drawing background")
    elif k == ord('1'):
        mode = 1
        print("Drawing foreground")
    elif k == ord('2'):
        mode = 2
        print("Drawing rectangle")
    elif k == ord('r'):
        mask = np.zeros(img.shape[:2], np.uint8)
        img_display = img.copy()
        rect = None
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        print("Reset")
    elif k == ord('g'):
        if rect is not None:
            rect = (min(ix, rect[0]), min(iy, rect[1]), abs(rect[2] - rect[0]), abs(rect[3] - rect[1]))
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img_display = img * mask2[:, :, np.newaxis]
        else:
            cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img_display = img * mask2[:, :, np.newaxis]
    elif k == 27:
        break

cv2.destroyAllWindows()
