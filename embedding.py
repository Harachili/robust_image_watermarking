import math
import time
from mpmath import csc
import os
import numpy as np
import cv2
import pywt
from scipy.signal import convolve2d
from math import sqrt
#from attacks import *

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

class colors():
    GREEN = "\033[92m"
    RED = "\033[91m"
    END = "\033[0m"


# get required images
if not os.path.isfile('lena.bmp'):  
  os.system('wget -O lena.bmp "https://drive.google.com/uc?export=download&id=17MVOgbOEOqeOwZCzXsIJMDzW5Dw9-8zm"')
if not os.path.isfile('csf.csv'):  
  os.system('wget -O csf.csv "https://drive.google.com/uc?export=download&id=1w43k1BTfrWm6X0rqAOQhrbX6JDhIKTRW"')

# define settings
#IMG_PATH='lena.bmp'
MARK_SIZE = 1024
BLOCK_SIZE = 4
ALPHA = 2

# Load the cover image
#image = cv2.imread(IMG_PATH, 0)

# Load the watermark
mark = np.load('you_shall_not_mark.npy')
mark = mark.reshape(32,32) # make the mark a 32x32 matrix


def quantize(imm, block_size=BLOCK_SIZE):
    # it creates a 32x32 4d matrix with 8x8 sub-matrices in A[i,j]. 
    # The 4d matrix is a 32x32 matrix because the original one is 128x128 and each block is 4x4
    return np.reshape(imm, (32, 32, block_size, block_size))


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
    Y = np.matmul(np.matmul(V,imm), V.T)
    return Y


def iapdcbt(Y):
    M, N = Y.shape
    V = np.zeros((M,N))
    for m in range(M):
        for n in range(N):
            if n == 0:
                V[m][n] = (N-m)/(N**2)
            else:
                V[m][n] = ((N-m)*math.cos((m*n*math.pi)/N) - csc((n*math.pi)/N)*math.sin((m*n*math.pi)/N))/(N**2)
    
    Vinv = np.linalg.inv(V) # matrice inversa
    X = Vinv@Y@Vinv.T
    return X


def hide_mark(image_after_apdcbt, mark, alpha=ALPHA): # 256x256, 32x32, ALPHA

    dimx,dimy = image_after_apdcbt.shape
    
    Y_vec = np.reshape(image_after_apdcbt,(dimx*dimy, ))
    Y_sgn = np.sign(Y_vec)
    Y_mod = np.abs(Y_vec)
    Y_index = np.argsort(-Y_mod, axis=None)

    #Embedding
    Yw_mod = Y_mod
    for idx, loc in enumerate(Y_index[1:1024+1]):
        Yw_mod[loc] = Y_mod[loc] + alpha * mark[idx]

    Y_new_vec = np.multiply(Yw_mod,Y_sgn)
    Y_new = np.reshape(Y_new_vec,(dimx, dimy))
    return Y_new

def get_coefficient_matrix(image_block):
    for i in range(32):
        for j in range(32):
            image_block[i][j] = apdcbt(image_block[i][j])
    return image_block

def embedd_into_sub_band(sub_band, mark, block_size=BLOCK_SIZE):
    image_block = quantize(sub_band, block_size) # divide the image in 16x16x8x8
    coefficient_matrices = get_coefficient_matrix(image_block)
    coefficient_matrices = np.reshape(coefficient_matrices, (128, 128))

    ##############################################
    embedded = hide_mark(coefficient_matrices, mark, alpha=ALPHA)
    embedded = np.reshape(embedded, (32, 32, block_size, block_size))
    ##############################################

    watermarked_image_block = np.zeros((32, 32, block_size, block_size))
    for i in range(32):
        for j in range(32):
            # la watermarked deve avere il DC coefficient della modified_coefficient_matrix e il resto dell'image block
            watermarked_image_block[i][j] = iapdcbt(embedded[i][j])
    
    sub_band_w = np.resize(watermarked_image_block, (128, 128))

    return sub_band_w

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

def embedding(image, mark):

    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL, 'haar')
    
    mark = np.reshape(mark, (1024,))

    LH2_w = embedd_into_sub_band(LH2, mark, block_size=BLOCK_SIZE)
    HL2_w = embedd_into_sub_band(HL2, mark, block_size=BLOCK_SIZE)


    LL_w = pywt.idwt2((LL2, (LH2_w, HL2_w, HH2)), 'haar')
    watermarked = pywt.idwt2((LL_w, (LH, HL, HH)), 'haar')
    
    watermarked = watermarked.astype(np.uint8)
    return watermarked

def merge_watermarks(w1, w2, t=.5):
    watermark = (w1 + w2)/2
    for i in range(32):
        for j in range(32):
            if watermark[i][j] >= t:
                watermark[i][j] = 1
            else:
                watermark[i][j] = 0
    return watermark


def check_wm(watermark_originale, watermark_attacked):
    watermark_originale = np.reshape(watermark_originale, (1024,))
    watermark_attacked = np.reshape(watermark_attacked, (1024,))
    sim = similarity(watermark_originale, watermark_attacked)
    T = 12.09
    if sim > T:
        return 1
    else:
        return 0

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    return s

def compute_thr(sim, mark_size, w):
    SIM = np.zeros(1000)
    SIM[0] = abs(sim)
    for i in range(1, 1000):
        r = np.random.uniform(0.0, 1.0, mark_size)
        SIM[i] = abs(similarity(w, r))
    
    SIM.sort()
    t = SIM[-2]
    T = t + (0.1*t)
    print('Threshold:', T)
    return T



if __name__ == "__main__":

    image = cv2.imread('lena.bmp', 0)
        
    watermarked = embedding(image, mark)
    cv2.imwrite('watermarked.bmp', watermarked)
    w = wpsnr(image, watermarked)
    print(f'wPSNR lena.bmp - watermarked.bmp: %.2fdB' % w)
