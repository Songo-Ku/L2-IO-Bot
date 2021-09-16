import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from functions import misc
from classes import Vision
from classes import Constants
import win32gui
import time


vision = Vision.Vision(-1)

img = 'full.png'
# tem = 'targ_cue.png'

im = cv2.imread(img,0)
x,y,w,h, = Constants.Constants.SCR_TARGCUE
# im = im[y:y+h, x:x+w]
# im = Image.fromarray(im)
# im.save('full.png','png')
# # im = np.array(im)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#cv2.imshow('image',im)
#cv2.waitKey()

#templ = cv2.imread(tem,0)
# templ = cv2.cvtColor(templ,cv2.COLOR_BGR2GRAY)

#cv2.imshow('template',templ)
#cv2.waitKey()

print vision.match_template(im, Constants.Constants.TMPL_TARGET_CUE)

