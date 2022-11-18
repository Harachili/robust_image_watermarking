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

#AWGN
def attack_0(img):
    img = cv2.imread(img, 0)
    mean = 0.0   # some constant
    #np.random.seed(seed)
    attacked = img + np.random.normal(mean, 1, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def attack_1(img):
    img = cv2.imread(img, 0)
    mean = 0.0   # some constant
    #np.random.seed(seed)
    attacked = img + np.random.normal(mean, 5, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def attack_2(img):
    img = cv2.imread(img, 0)
    mean = 0.0   # some constant
    #np.random.seed(seed)
    attacked = img + np.random.normal(mean, 10, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def attack_3(img):
    img = cv2.imread(img, 0)
    mean = 0.0   # some constant
    #np.random.seed(seed)
    attacked = img + np.random.normal(mean, 15, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def attack_4(img):
    img = cv2.imread(img, 0)
    mean = 0.0   # some constant
    #np.random.seed(seed)
    attacked = img + np.random.normal(mean, 20, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

#BLUR
def attack_5(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 0.25)
    return attacked

def attack_6(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 0.5)
    return attacked

def attack_7(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 1)
    return attacked

def attack_8(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 2)
    return attacked

#MEDIAN
def attack_9(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [3,1])
    return attacked

def attack_10(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [1,3])
    return attacked

def attack_10(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [3,3])
    return attacked

def attack_11(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [5,3])
    return attacked

def attack_12(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [3,5])
    return attacked

def attack_13(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [5,7])
    return attacked

def attack_14(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [7,7])
    return attacked

#JPEG
def attack_15(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=90)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_16(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=80)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_17(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=70)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_18(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=60)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_19(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=50)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_20(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=40)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_21(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=30)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_22(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=20)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_23(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=10)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_24(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=5)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_25(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=3)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_26(img):
    img = cv2.imread(img, 0)
    img = Image.fromarray(img)
    img.save('tmp.jpg',"JPEG", quality=2)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

#BLUR - MEDIAN
def attack_27(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 1)
    attacked = medfilt(attacked, [3,3])
    return attacked

def attack_28(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 1)
    attacked = medfilt(attacked, [5,5])
    return attacked

def attack_29(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 2)
    attacked = medfilt(attacked, [5,5])
    return attacked

#BLUR - MEDIAN - JPEG
def attack_30(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, .5)
    attacked = medfilt(attacked, [3,3])
    img = Image.fromarray(attacked)
    img.save('tmp.jpg',"JPEG", quality=30)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_31(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 1)
    attacked = medfilt(attacked, [3,3])
    img = Image.fromarray(attacked)
    img.save('tmp.jpg',"JPEG", quality=20)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_32(img):
    img = cv2.imread(img, 0)
    attacked = gaussian_filter(img, 1)
    attacked = medfilt(attacked, [5,5])
    img = Image.fromarray(attacked)
    img.save('tmp.jpg',"JPEG", quality=20)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

#MEDIAN - JPEG
def attack_33(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [3,3])
    img = Image.fromarray(attacked)
    img.save('tmp.jpg',"JPEG", quality=30)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_34(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [3,3])
    img = Image.fromarray(attacked)
    img.save('tmp.jpg',"JPEG", quality=10)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_35(img):
    img = cv2.imread(img, 0)
    attacked = medfilt(img, [5,5])
    img = Image.fromarray(attacked)
    img.save('tmp.jpg',"JPEG", quality=10)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

#RESIZE - JPEG
def attack_36(img):
    img = cv2.imread(img, 0)
    x, y = img.shape
    attacked = rescale(img, .8)
    attacked = rescale(attacked, 1/.8)
    attacked = attacked[:x, :y] * 255

    img = Image.fromarray(attacked.astype(np.uint8))
    img.save('tmp.jpg',"JPEG", quality=50)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_37(img):
    img = cv2.imread(img, 0)
    x, y = img.shape
    attacked = rescale(img, .8)
    attacked = rescale(attacked, 1/.8)
    attacked = attacked[:x, :y] * 255

    img = Image.fromarray(attacked.astype(np.uint8))
    img.save('tmp.jpg',"JPEG", quality=30)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

def attack_38(img):
    img = cv2.imread(img, 0)
    x, y = img.shape
    attacked = rescale(img, .6)
    attacked = rescale(attacked, 1/.6)
    attacked = attacked[:x, :y] * 255

    img = Image.fromarray(attacked.astype(np.uint8))
    img.save('tmp.jpg',"JPEG", quality=10)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked,dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

#AWGN - MEDIAN
def attack_39(img):
    img = cv2.imread(img, 0)
    mean = 0.0
    attacked = img + np.random.normal(mean, 5, img.shape)
    attacked = np.clip(attacked, 0, 255)
    attacked = medfilt(attacked, [3,3])
    return attacked

def attack_40(img):
    img = cv2.imread(img, 0)
    mean = 0.0
    attacked = img + np.random.normal(mean, 10, img.shape)
    attacked = np.clip(attacked, 0, 255)
    attacked = medfilt(attacked, [3,3])
    return attacked

def attack_41(img):
    img = cv2.imread(img, 0)
    mean = 0.0
    attacked = img + np.random.normal(mean, 15, img.shape)
    attacked = np.clip(attacked, 0, 255)
    attacked = medfilt(attacked, [3,5])
    return attacked

def attack_42(img):
    img = cv2.imread(img, 0)
    mean = 0.0
    attacked = img + np.random.normal(mean, 15, img.shape)
    attacked = np.clip(attacked, 0, 255)
    attacked = medfilt(attacked, [5,5])
    return attacked

def attack_43(img, kernel_size=[3,3]): # low_center_median_med
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 3
    loc_end_row = h * 2 // 3
    loc_start_col = w // 3
    loc_end_col = w * 2 // 3

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_44(img, sigma=2): # low_center_blur_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 3
    loc_end_row = h * 2 // 3
    loc_start_col = w // 3
    loc_end_col = w * 2 // 3

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_45(img, sigma=.25): # low_center_blur_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 3
    loc_end_row = h * 2 // 3
    loc_start_col = w // 3
    loc_end_col = w * 2 // 3

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_46(img, sigma=1): # low_center_blur_med_HIGH
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 3
    loc_end_row = h * 2 // 3
    loc_start_col = w // 3
    loc_end_col = w * 2 // 3

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_47(img, sigma=.5): # low_center_blur_med_LOW
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 3
    loc_end_row = h * 2 // 3
    loc_start_col = w // 3
    loc_end_col = w * 2 // 3

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_48(img, kernel_size = [3,1]): # low_lower_left_median_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_49(img, kernel_size = [5,5]): # low_lower_left_median_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_50(img, kernel_size=[3,3]): # low_lower_left_median_med
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_51(img, sigma=2): # low_upper_center_blur_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = w // 2
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_52(img, sigma=.25): # low_lower_right_blur_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = w // 2
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_53(img, sigma=1): # low_lower_right_blur_med_HIGH
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = w // 2
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_54(img, sigma=.5): # low_lower_right_blur_med_LOW
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = w // 2
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_55(img, kernel_size = [3,1]): # low_lower_right_median_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = w // 2
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_56(img, kernel_size = [5,5]): # low_lower_right_median_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = w // 2
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_57(img, kernel_size=[3,3]): # low_lower_right_median_med
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = w // 2
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_58(img, sigma=2): # low_upper_center_blur_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_59(img, sigma=.25): # low_upper_center_blur_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_60(img, sigma=1): # low_upper_center_blur_med_HIGH
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_61(img, sigma=.5): # low_upper_center_blur_med_LOW
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_62(img, kernel_size = [3,1]): # low_upper_center_median_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_63(img, kernel_size = [5,5]): # low_upper_center_median_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_64(img, kernel_size=[3,3]): # low_upper_center_median_med
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_65(img, sigma=2): # low_upper_right_blur_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_66(img, sigma=.25): # low_upper_right_blur_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_67(img, sigma=1): # low_upper_right_blur_med_HIGH
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_68(img, sigma=.5): # low_upper_right_blur_med_LOW
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_69(img, kernel_size = [3,1]): # low_upper_left_median_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_70(img, kernel_size = [5,5]): # low_upper_left_median_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_71(img, kernel_size=[3,3]): # low_upper_left_median_med
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = h // 2
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_72(img, sigma=2): # low_upper_right_blur_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = np.round(h // 2)
    loc_start_col = np.round(w // 2)
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_73(img, sigma=.25): # low_upper_right_blur_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = np.round(h // 2)
    loc_start_col = np.round(w // 2)
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_74(img, sigma=1): # low_upper_right_blur_med_HIGH
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = np.round(h // 2)
    loc_start_col = np.round(w // 2)
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_75(img, sigma=.5): # low_upper_right_blur_med_LOW
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = np.round(h // 2)
    loc_start_col = np.round(w // 2)
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_76(img, kernel_size = [3,1]): # low_upper_right_median_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = np.round(h // 2)
    loc_start_col = np.round(w // 2)
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_77(img, kernel_size = [5,5]): # low_upper_right_median_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = np.round(h // 2)
    loc_start_col = np.round(w // 2)
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_78(img, kernel_size = [3,3]): # low_upper_right_median_mid
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = 1
    loc_end_row = np.round(h // 2)
    loc_start_col = np.round(w // 2)
    loc_end_col = w

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_79(img, kernel_size = [3,1]): # median_low
    img = cv2.imread(img, 0)
    attacked = medfilt(img, kernel_size)
    return attacked

def attack_80(img, kernel_size = [5,5]): # median_high
    img = cv2.imread(img, 0)
    attacked = medfilt(img, kernel_size)
    return attacked

def attack_81(img, kernel_size = [3,3]): # median_mid
    img = cv2.imread(img, 0)
    attacked = medfilt(img, kernel_size)
    return attacked

def attack_82(img, scale = .5): # resize_high
    img = cv2.imread(img, 0)
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1/scale)
    attacked = attacked[:x, :y]
    return attacked*255

def attack_83(img, scale = .9): # resize_low
    img = cv2.imread(img, 0)
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1/scale)
    attacked = attacked[:x, :y]
    return attacked*255

def attack_84(img, scale = .75): # resize_mid
    img = cv2.imread(img, 0)
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1/scale)
    attacked = attacked[:x, :y]
    return attacked*255

def attack_85(img, sigma = 1.5, alpha = 1.2): # sharpening_strong
    img = cv2.imread(img, 0)
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def attack_86(img, sigma = .6, alpha = .05): # sharpening_mid
    img = cv2.imread(img, 0)
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked
    
def attack_87(img, sigma=2): # low_lower_left_blur_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_88(img, sigma=.25): # low_lower_left_blur_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_89(img, sigma=1): # low_lower_left_blur_med_HIGH
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_90(img, sigma=.5): # low_lower_left_blur_med_LOW
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 2
    loc_end_row = h
    loc_start_col = 1
    loc_end_col = w // 2

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = gaussian_filter(to_attack, sigma)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_91(img, kernel_size = [3,1]): # low_center_median_low
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 3
    loc_end_row = h * 2 // 3
    loc_start_col = w // 3
    loc_end_col = w * 2 // 3

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt

def attack_92(img, kernel_size = [5,5]): # low_center_median_high
    img = cv2.imread(img, 0)
    h,w = img.shape
    loc_start_row = h // 3
    loc_end_row = h * 2 // 3
    loc_start_col = w // 3
    loc_end_col = w * 2 // 3

    to_attack = img[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1]
    attacked = medfilt(to_attack, kernel_size)
    Iatt = img
    Iatt[loc_start_row:loc_end_row+1,loc_start_col:loc_end_col+1] = attacked
    return Iatt




attacked = attack_86('watermarked_images/tree_youshallnotmark.bmp')

watermarked = cv2.imread('watermarked_images/tree_youshallnotmark.bmp', 0)
w = wpsnr(watermarked, attacked)
print('wPSNR watermarked - attacked: %.2fdB' % w)

cv2.imwrite('attacked_images/youshallnotmark_theyarethesamepicture_tree.bmp', attacked)