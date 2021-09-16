import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from functions import misc
from classes import Vision
from classes import Constants

# name?


def method_1(filename):
    im = Image.open(filename + ".png")  # the second one
    #
    enhancer = ImageEnhance.Contrast(im)
    # im = im.filter(ImageFilter.)
    im = enhancer.enhance(5)
    # im = im.filter(ImageFilter.GaussianBlur(0.1))

    # im = im.convert('1')
    im.save(filename+'_method1'+'.png')
    text = pytesseract.image_to_string(Image.open(filename+'_method1'+'.png'),lang='l2fnt3')
    print(text)


def get_char_mp(full_img):

    #im = cv2.imread(filename)
    x_, y_, w_, h_ = Constants.Constants.SCR_MP
    im = full_img[y_:y_ + h_, x_:x_ + w_]
    # enhancer = ImageEnhance.Contrast(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = np.array(img2gray)
    # do stuff with im, then save suffixed _method1

    # im = im.filter(ImageFilter.)


    # img2gray = img
    # ret, mask = cv2.threshold(im, 60, 250, cv2.THRESH_BINARY)
    ret, mask = cv2.threshold(im, 135, 255, cv2.THRESH_BINARY) # HP
    image_final = cv2.bitwise_and(im, im, mask=mask)
    # image_final = cv2.resize(image_final, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    Image.fromarray(image_final).save('mp.png')
    text = pytesseract.image_to_string(Image.open('mp.png'), lang='l2fnt', config='--oem 3 --psm 6 outputbase l2_hp').encode('ascii','ignore')
    print(text)
    #text = text.replace('/', ' ')
    #life = map(str.split, text.split('\n'))
    #print life
    #cp = map(int, life[0])
    #hp = map(int, life[1])
    #mp = map(int, life[2])



def method_3(filename):
    print Vision.Vision.txt_from_img()



im = cv2.imread('full.bmp')

print 'start'
get_char_mp(im)
get_char_mp(im)
get_char_mp(im)
print 'end'
