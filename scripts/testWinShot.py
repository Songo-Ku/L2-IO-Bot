import numpy
import cv2
import time
import msvcrt

from win32gui import GetForegroundWindow
from classes import Constants
import pytesseract as pytesseract

from classes import Vision
from PIL import Image
import win32con

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'

flame = cv2.imread('flame.png', 0)
flame = numpy.array(flame)
flame = Image.fromarray(flame)
flame.save('greyflame.png')

global vision, win


ch = raw_input("on start: ")

if ch == 'start':
    timer = 3
    while timer > 0:
        print "Getting window in {0} ... ".format(timer)
        timer -= 1
        time.sleep(1)

    vision = Vision.Vision()
    win = GetForegroundWindow()

ch2 = raw_input("on shot")
if ch2 == 'shot':
    img_ = vision.getWinPart(win,Constants.Constants.SCR_CHAT)

    # vision.txtimg_clean('screen_part.png')

    # print(pytesseract.image_to_string(cleaned_))
    vision.captch_ex('screen_part.png',(80,200),(60,200))
