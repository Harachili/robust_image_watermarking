import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from PIL import Image
import cv2

if not os.path.isfile('csf.csv'):  
  os.system('wget -O csf.csv "https://drive.google.com/uc?export=download&id=1w43k1BTfrWm6X0rqAOQhrbX6JDhIKTRW"')


from detection import wpsnr



def awgn(img, std, seed):
    mean = 0.0   # some constant
    #np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

def sharpening(img, sigma, alpha):
    #print(img/255)
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked

def resizing(img, scale):
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1/scale)
    attacked = attacked[:x, :y]
    return attacked

def jpeg_compression(img, QF):
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=QF)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked



watermarked = cv2.imread('watermarked.bmp', 0)

############## Possible attacks ##############
#attacked = watermarked #no attacks performed
#attacked = blur(watermarked, [1, 1])
#attacked = sharpening(watermarked, 1, 1)
#attacked = median(watermarked, [3, 5])
#attacked = awgn(watermarked, 30.0, 123)
#attacked = resizing(watermarked, 0.5)
attacked = jpeg_compression(watermarked, 50)
##############################################

w = wpsnr(watermarked, attacked)
print('wPSNR watermarked - attacked: %.2fdB' % w)

cv2.imwrite('attacked.bmp', attacked)