import math
import time
from matplotlib import pyplot as plt
from mpmath import csc
import numpy as np
import cv2
import pywt
from scipy.signal import convolve2d
from math import sqrt
from sklearn.preprocessing import normalize

class colors():
    GREEN = "\033[92m"
    RED = "\033[91m"
    END = "\033[0m"

BLOCK_SIZE = 4
ALPHA = 2
THRESHOLD = 11.79

# it creates a 16x16 4d matrix with 8x8 sub-matrices in A[i,j]. The 4d matrix is a 16x16 matrix because the original one is 128x128 and each block is 8x8
def quantize(imm, block_size=BLOCK_SIZE):
    return np.reshape(imm, (16, 16, block_size, block_size))

def apdcbt(imm):
    M, N = imm.shape
    Y = np.zeros((M,N))
    V = np.zeros((M,N))
    for m in range(M):
        for n in range(N):
            if n == 0:
                V[m][n] = (N-m)/(N**2)
            else:
                V[m][n] = ((N-m)*math.cos((m*n*math.pi)/N) - csc((n*math.pi)/N)*math.sin((m*n*math.pi)/N))/(N**2)
    Y = V@imm@V.T
    return Y
    
def get_coefficient_matrix(image_block):
    for i in range(32):
        for j in range(32):
            image_block[i][j] = apdcbt(image_block[i][j])
    return image_block


def merge_watermarks(w1, w2, t=.5):
    watermark = (w1 + w2)/2
    for i in range(32):
        for j in range(32):
            if watermark[i][j] >= t:
                watermark[i][j] = 1
            else:
                watermark[i][j] = 0
    return watermark

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    return s

def check_wm(watermark_originale, watermark_attacked):
    watermark_originale = np.reshape(watermark_originale, (1024,))
    watermark_attacked = np.reshape(watermark_attacked, (1024,))
    if watermark_attacked.any() == 1:
        sim = similarity(watermark_originale, watermark_attacked)
    else:
        sim = 0
    if sim > THRESHOLD:
        return 1
    else:
        return 0

def extract_watermark(image, watermarked, alpha=ALPHA):

    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL, 'haar')

    LL_w, (LH_w, HL_w, HH_w) = pywt.dwt2(watermarked, 'haar')
    LL2_w, (LH2_w, HL2_w, HH2_w) = pywt.dwt2(LL_w, 'haar')

    LH2 = np.reshape(LH2, (32, 32, 4, 4))
    HL2 = np.reshape(HL2, (32, 32, 4, 4))

    coefficient_matrix_LH2 = get_coefficient_matrix(LH2)
    coefficient_matrix_HL2 = get_coefficient_matrix(HL2)

    LH2_w = np.reshape(LH2_w, (32, 32, 4, 4))
    HL2_w = np.reshape(HL2_w, (32, 32, 4, 4))

    coefficient_matrix_LH2_w = get_coefficient_matrix(LH2_w)
    coefficient_matrix_HL2_w = get_coefficient_matrix(HL2_w)


    w1 = np.zeros(1024, dtype=np.float64)
    w2 = np.zeros(1024, dtype=np.float64)

    # extract one watermark from horizontal component
    coefficient_matrix_LH2 = abs(coefficient_matrix_LH2)
    locations = np.argsort(-coefficient_matrix_LH2, axis=None)

    coefficient_matrix_LH2_w = abs(coefficient_matrix_LH2_w)

    coefficient_matrix_LH2 = np.reshape(coefficient_matrix_LH2, (32*32*4*4,))
    coefficient_matrix_LH2_w = np.reshape(coefficient_matrix_LH2_w, (32*32*4*4,))

    for idx, loc in enumerate(locations[1:1024+1]):
        w1[idx] =  (coefficient_matrix_LH2_w[loc] - coefficient_matrix_LH2[loc]) / alpha

    # extract identical watermark from vertical component
    coefficient_matrix_HL2 = abs(coefficient_matrix_HL2)
    locations = np.argsort(-coefficient_matrix_HL2, axis=None)

    coefficient_matrix_HL2_w = abs(coefficient_matrix_HL2_w)

    coefficient_matrix_HL2 = np.reshape(coefficient_matrix_HL2, (32*32*4*4,))
    coefficient_matrix_HL2_w = np.reshape(coefficient_matrix_HL2_w, (32*32*4*4,))

    for idx, loc in enumerate(locations[1:1024+1]):
        w2[idx] =  (coefficient_matrix_HL2_w[loc] - coefficient_matrix_HL2[loc]) / alpha

    w1 = np.reshape(w1, (32, 32))
    w2 = np.reshape(w2, (32, 32))
    
    return w1, w2


def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0

  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  csf = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels

def detection(image, watermarked, attacked):
    
    ######## comment before running ROCCurve.py, uncomment before running tester.py
    image = cv2.imread(image, 0)
    watermarked = cv2.imread(watermarked, 0)
    attacked = cv2.imread(attacked, 0)
    ########

    W1, W2 = extract_watermark(image, watermarked)
    original_watermark = merge_watermarks(W1, W2).astype(np.uint8)

    W1, W2 = extract_watermark(image, attacked)
    extracted_watermark = merge_watermarks(W1, W2).astype(np.uint8)
    
    wm_found = check_wm(original_watermark, extracted_watermark)
    
    wpsnr_wat_att = wpsnr(watermarked, attacked)

    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.title('Original WM')
    plt.imshow(original_watermark, cmap='gray')
    plt.subplot(122)
    plt.title('Extracted WM')
    plt.imshow(extracted_watermark, cmap='gray')
    plt.show()

    return wm_found, wpsnr_wat_att, extracted_watermark

if __name__ == "__main__":

    wm_found, wpsnr_wat_att, extracted_watermark = detection('lena.bmp', 'watermarked.bmp', 'attacked.bmp')

    if wm_found == 1:
        print(f'{colors.GREEN}Mark has been found{colors.END}\nwPSNR: %.2fdB' % wpsnr_wat_att)
    else:
        print(f'{colors.RED}Mark has been lost{colors.END}\nwPSNR: %.2fdB' % wpsnr_wat_att)
