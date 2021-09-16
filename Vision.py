import numpy

import cv2
import win32gui
import win32ui
import re
import unicodedata
from ctypes import windll

import win32con
from mss import mss
from PIL import Image
import numpy as np
import time

from pytesseract import pytesseract

from classes.Constants import Constants


pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'

debug = False
ocr_ind = 0

class Vision:
    data = {'char': {}, 'target': {}, 'adena': -1, 'exp': -1, 'chat': [], 'mem': [], 'pt':[]}
    char = {}
    target = {}
    adena = 0
    exp = 0.0
    chat = []
    mem = []
    pt = []
    snapshot = []
    winw = 1600
    winh=900

    def __init__(self,hwnd_):
        self.hwnd = hwnd_
        # self.static_templates = {
        #     'left-goalpost': 'assets/left-goalpost.png',
        #     'bison-head': 'assets/bison-head.png',
        #     'pineapple-head': 'assets/pineapple-head.png',
        #     'bison-health-bar': 'assets/bison-health-bar.png',
        #     'pineapple-health-bar': 'assets/pineapple-health-bar.png',
        #     'cancel-button': 'assets/cancel-button.png',
        #     'filled-with-goodies': 'assets/filled-with-goodies.png',
        #     'next-button': 'assets/next-button.png',
        #     'tap-to-continue': 'assets/tap-to-continue.png',
        #     'unlocked': 'assets/unlocked.png',
        #     'full-rocket': 'assets/full-rocket.png'
        # }
        #
        # self.templates = { k: cv2.imread(v, 0) for (k, v) in self.static_templates.items() }
        #
        # self.monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        # self.screen = mss()
        #
        # self.frame = None

    def update_short(self):
        self.get()
        self.update_char()
        self.update_target()
        print 'AGENT update_short into:'
        print self.char
        print self.target
        print '----------------------------'

    def update_full(self):
        self.get()
        self.update_char()
        self.update_charname()
        self.update_lvl()
        self.update_exp()
        self.update_adena()
        self.update_target()
        # self.data['char'] = self.char
        # self.data['target'] = self.target
        # self.data['adena'] = self.adena
        # self.data['exp'] = self.exp
        # self.data['chat'] = self.chat
        # self.data['mem'] = self.mem
        # self.data['pt'] = self.pt
        print 'AGENT update_full into:'
        print self.char
        print self.lvl
        print self.exp
        print self.adena
        print self.target
        print '----------------------------'

    def convert_rgb_to_bgr(self, img):
        return img[:, :, ::-1]

    def take_screenshot(self):
        sct_img = self.screen.grab(self.monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        img = np.array(img)
        img = self.convert_rgb_to_bgr(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img_gray

    def refresh_frame(self):
        self.frame = self.take_screenshot()

    def match_template(self, img_grayscale, template, threshold=0.9):
        """
        Matches template image in a target grayscaled image
        """

        res = cv2.matchTemplate(img_grayscale, template, cv2.TM_CCOEFF_NORMED)
        matches = np.where(res >= threshold)
        return matches

    def find_template(self, name, image=None, threshold=0.9):
        if image is None:
            if self.frame is None:
                self.refresh_frame()

            image = self.frame

        return self.match_template(
            image,
            self.templates[name],
            threshold
        )

    def scaled_find_template(self, name, image=None, threshold=0.9, scales=[1.0, 0.9, 1.1]):
        if image is None:
            if self.frame is None:
                self.refresh_frame()

            image = self.frame

        initial_template = self.templates[name]
        for scale in scales:
            scaled_template = cv2.resize(initial_template, (0, 0), fx=scale, fy=scale)
            matches = self.match_template(
                image,
                scaled_template,
                threshold
            )
            if np.shape(matches)[1] >= 1:
                return matches
        return matches

    def img_from_win(self, debug=False):
        hwnd = win32gui.FindWindow(None, 'Lineage II')

        # Change the line below depending on whether you want the whole window
        # or just the client area.
        left, top, right, bot = win32gui.GetClientRect(hwnd)
        # left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, self.winw, self.winh)

        saveDC.SelectObject(saveBitMap)

        # Change the line below depending on whether you want the whole window
        # or just the client area.
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
        # result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
        # print result

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)




        # # hwnd = win32gui.FindWindow(None, 'Calculator')
        # hwnd_ = self.hwnd
        #
        # # Change the line below depending on whether you want the whole window
        # # or just the client area.
        # left, top, right, bot = win32gui.GetClientRect(hwnd_)
        # # left, top, right, bot = win32gui.GetWindowRect(hwnd_)
        #
        # xx_ = 0
        # yy_ = 0
        # w_ = 1280 # right - left
        # h_ = 720 # bot - top
        #
        # hwndDC = win32gui.GetWindowDC(hwnd_)
        # mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        # saveDC = mfcDC.CreateCompatibleDC()
        #
        # saveBitMap = win32ui.CreateBitmap()
        # saveBitMap.CreateCompatibleBitmap(mfcDC, w_, h_)
        #
        # # saveBitMap.CreateCompatibleBitmap(mfcDC, w_, h_)
        # saveDC.SelectObject(saveBitMap)
        # result = windll.user32.PrintWindow(hwnd_, saveDC.GetSafeHdc(), 0)
        # # saveDC.BitBlt((0, 0), (w_, h_), mfcDC, (xx_, yy_), win32con.SRCCOPY)
        #
        # bmpinfo = saveBitMap.GetInfo()
        # bmpstr = saveBitMap.GetBitmapBits(True)
        #
        # im = Image.frombuffer(
        #     'RGB',
        #     (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        #     bmpstr, 'raw', 'BGRX', 0, 1)
        #
        # win32gui.DeleteObject(saveBitMap.GetHandle())
        # saveDC.DeleteDC()
        # mfcDC.DeleteDC()
        # win32gui.ReleaseDC(hwnd_, hwndDC)
        #
        # # if result == 1:
        # # PrintWindow Succeeded
        #
        #
        # # MODDED: PART
        # # img = img[y_:y_ + h_, x_:x_ + w_]
        #
        # # img = np.array(im)
        # # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # img_gray = Image.fromarray(img_gray)
        # # time.sleep(1)
        # # nw = int(w_ * 2)
        # # nh = int(h_ * 2)
        # # sz = (nw, nh)
        # # im.thumbnail(sz, Image.ANTIALIAS)
        if debug:
            im.save("screen.png")

        # img_gray.thumbnail(sz)

        # im = np.array(im)
        return im
        # end if result==1

    # DEPRECATED
    def txtimg_clean(self,img_):
        img_ = cv2.imread(img_)
        # img_ = np.array(img_)

        cv2.imshow('normal', img_)
        cv2.waitKey()

        img2gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

        cv2.imshow('gray', img2gray)
        cv2.waitKey()

        ret, mask = cv2.threshold(img2gray, 150, 200, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)

        cv2.imshow('masked', image_final)
        cv2.waitKey()

        ret, new_img = cv2.threshold(image_final, 100, 200, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV

        cv2.imshow('tresholded', new_img)
        cv2.waitKey()

        sav_img = Image.fromarray(new_img,'L')
        sav_img.save('cleaned.png')
        # return new_img

    # DEPRECATED
    def text_from_win(self,hwnd_,(x_,y_,w_,h_), (min_,max_),(min2,max2), find_str=''):
        found_ = False
        # hwnd_ = self.hwnd
        hwndDC = win32gui.GetWindowDC(hwnd_)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w_, h_)

        saveDC.SelectObject(saveBitMap)

        saveDC.BitBlt((0, 0), (w_, h_), mfcDC, (x_, y_), win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd_, hwndDC)

        img = numpy.array(im)
        # img_final = cv2.imread(file_name)
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img2gray = img
        ret, mask = cv2.threshold(img2gray, min_, max_, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)

        # cv2.imshow('masked', image_final)
        # cv2.waitKey()

        ret, new_img = cv2.threshold(image_final, min2, max2,
                                     cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV

        # cv2.imshow('tresholded', new_img)
        # cv2.waitKey()

        '''
                line  8 to 12  : Remove noisy portion 
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 1))  # to manipulate the orientation of dilution,
        # large x means horizonatally dilating  more, large y means vertically dilating more
        dilated = cv2.dilate(new_img, kernel, iterations=3)  # dilate , more the iteration more the dilation

        # cv2.imshow('dilated', dilated)
        # cv2.waitKey()

        # for cv2.x.x

        _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # findContours returns 3 variables for getting contours

        # for cv3.x.x comment above line and uncomment line below

        # image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        out_array = []
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)
            # out_array.append([x,y,w,h])
            # CORRECTIONS
            y -= 2

            # Don't plot small false positives that aren't text
            if w < 35 and h < 35:
                continue

            # draw rectangle around contour on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # you can crop image and send to OCR  , false detected will return no text :)
            cropped = image_final[y:y + h, x: x + w]

            txt = pytesseract.image_to_string(cropped, config='--psm 7')  # .encode('ascii','ignore')
            if find_str == '':
                if len(txt) > 0:
                    print 'found white text: {0}'.format(txt)
                    found_ = True
            elif find_str in txt:
                print 'found [{0}] in [{1}]'.format(find_str, txt)
                found_ = True
            # out_array.append(pytesseract.image_to_string(cropped, config='--psm 7'))
            # out_array.append(pytesseract.image_to_string(cropped,config='--psm 7').encode('ascii','ignore'))

            # s = file_name + '/crop_' + str(index) + '.jpg'
            # cv2.imwrite(s , cropped)
            # index = index + 1

        # EXTRA TEST
        # print(pytesseract.image_to_string(image_final, config='--psm 4 --oem 3'))

        # write original image with added contours to disk
        sv = Image.fromarray(img)
        sv.save('test_bounds.png')
        # cv2.imshow('captcha_result', img)

        return found_

    def findChat(self,find_str=''):
        return self.findText(Constants.SCR_CHAT,(80,200),(80,200),find_str=find_str)

    def isSaidLocal(self):
        return self.findText(Constants.SCR_CHAT,(80,200),(150,200))

    def findSys(self,find_str=''):
        return self.findText(Constants.SCR_SYSMSG, (80, 200), (80, 200), find_str=find_str)

    def text_from_img(self, img, (min,max),(min2,max2), find_str='', charset=Constants.CHARSET_NUM, shape = '',debug=False):
        mode=""
        if shape=='word':
            mode=" --psm 8"
        elif shape=='line':
            mode=" --psm 7"
        elif shape=='block':
            mode=" --psm 5"

        cfg_str = "-c tessedit_char_whitelist=0123456789/ --oem 3 --psm 8"
        # cfg_str = "-c tessedit_char_whitelist="+charset+mode+" --psm 5"
        if debug:
            print cfg_str

        found_ = False
        # img = cv2.imread(file_name)
        img_final = numpy.array(img)
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img2gray = img
        ret, mask = cv2.threshold(img2gray, min, max, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)

        if debug:
            cv2.imshow('masked', image_final)
            cv2.waitKey()

        ret, new_img = cv2.threshold(image_final, min2, max2, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV

        if debug:
            cv2.imshow('tresholded', new_img)
            cv2.waitKey()

        '''
                line  8 to 12  : Remove noisy portion 
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 2))  # to manipulate the orientation of dilution,
        # large x means horizonatally dilating  more, large y means vertically dilating more
        dilated = cv2.dilate(new_img, kernel, iterations=1)  # dilate , more the iteration more the dilation

        if debug:
            cv2.imshow('dilated', dilated)
            cv2.waitKey()

        # for cv2.x.x

        _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # findContours returns 3 variables for getting contours

        # for cv3.x.x comment above line and uncomment line below

        # image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # out_array = []
        index=1

        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)
            # out_array.append([x,y,w,h])
            # CORRECTIONS
            y -= 1

            # Don't plot small false positives that aren't text
            if w < 35 and h < 35:
                continue

            # draw rectangle around contour on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)

            # you can crop image and send to OCR  , false detected will return no text :)
            cropped = image_final[y:y + h, x: x + w]

            if debug:
                savimg = Image.fromarray(cropped)
                savimg.save("cropped_{0}.png".format(index))

            txt = pytesseract.image_to_string(cropped, config=cfg_str).encode('ascii','ignore')
            if len(txt)>0:
                print '[{0}]  {1}'.format(str(index), txt)
                found_ = True
                if find_str != '':
                    if find_str in txt:
                        print 'found [{0}] in [{1}]'.format(find_str,txt)

            # out_array.append(pytesseract.image_to_string(cropped, config='--psm 7'))
            # out_array.append(pytesseract.image_to_string(cropped,config='--psm 7').encode('ascii','ignore'))

            # s = file_name + '/crop_' + str(index) + '.jpg'
            # cv2.imwrite(s , cropped)
            index = index + 1

        # EXTRA TEST
        # print(pytesseract.image_to_string(image_final, config='--psm 4 --oem 3'))
        if debug:
            cv2.imshow('captcha_result', img)
        # write original image with added contours to disk
        if debug:
            sv = Image.fromarray(img)
            sv.save('test_bounds.png')

        return found_

    def txt_from_img(self, img, (min,max),(min2,max2), find_str='', cfg_str = "--oem 3 --psm 7", dil_or = (9, 3), itr_ = 2, yshift = -2,debug=False,fac=2):
        # debug=True
        txt_ = []

        ''' 7 mostly
        mode=""
        if shape=='word':
            mode=" --psm 8"
        elif shape=='line':
            mode=" --psm 7"
        elif shape=='block':
            mode=" --psm 5"
        '''

        # cfg_str = "-c tessedit_char_whitelist=0123456789/ --oem 3 --psm 8"
        # cfg_str = "-c tessedit_char_whitelist="+charset+mode+" --psm 5"

        found_ = False
        # img = cv2.imread(file_name)
        img_final = numpy.array(img)
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img2gray = img
        ret, mask = cv2.threshold(img2gray, min, max, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)

        if debug:
            cv2.imshow('masked', image_final)
            cv2.waitKey()

        ret, new_img = cv2.threshold(image_final, min2, max2, cv2.THRESH_BINARY)
        # for black text , cv.THRESH_BINARY_INV

        if debug:
            cv2.imshow('tresholded', new_img)
            cv2.waitKey()

        '''
                line  8 to 12  : Remove noisy portion 
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, dil_or)  # to manipulate the orientation of dilution,
        # large x means horizonatally dilating  more, large y means vertically dilating more
        dilated = cv2.dilate(new_img, kernel, iterations=itr_)  # dilate , more the iteration more the dilation

        if debug:
            cv2.imshow('dilated', dilated)
            cv2.waitKey()

        # for cv2.x.x

        _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # findContours returns 3 variables for getting contours

        # for cv3.x.x comment above line and uncomment line below

        # image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # out_array = []
        index=1

        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)
            # out_array.append([x,y,w,h])
            # CORRECTIONS
            if y+yshift >=0 and y+yshift<h:
                y += yshift

            # Don't plot small false positives that aren't text
            if w < 2 and h < 2:
                continue

            # draw rectangle around contour on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)

            # you can crop image and send to OCR  , false detected will return no text :)
            cropped = image_final[y:y + h, x: x + w]

            # cv2.waitKey()
            if debug:
                cv2.imshow('cropped', cropped)
                cv2.waitKey()
            # toscale = Image.fromarray(cropped)
            scaled = cv2.resize(cropped, (0, 0), fx=fac, fy=fac,interpolation=cv2.INTER_LINEAR)

            if debug:
                savimg = Image.fromarray(scaled)
                global ocr_ind
                savimg.save("OCR_{0}.png".format(ocr_ind))
                ocr_ind += 1

            txt = pytesseract.image_to_string(scaled, config=cfg_str).encode('ascii','ignore')

            if len(txt)>0:
                txt_.append(txt)
                # print '[{0}]  {1}'.format(str(index), txt)

                if find_str != '':
                    if find_str in txt:
                        print 'found [{0}] in [{1}]'.format(find_str,txt)
                        found_ = True

            # out_array.append(pytesseract.image_to_string(cropped, config='--psm 7'))
            # out_array.append(pytesseract.image_to_string(cropped,config='--psm 7').encode('ascii','ignore'))

            # s = file_name + '/crop_' + str(index) + '.jpg'
            # cv2.imwrite(s , cropped)
            index = index + 1

        # print 'CP: {0} / {1}'.format(cp[0],cp[1])
        # print 'HP: {0} / {1}'.format(hp[0], hp[1])
        # print 'MP: {0} / {1}'.format(mp[0], mp[1])

        # EXTRA TEST
        # print(pytesseract.image_to_string(image_final, config='--psm 4 --oem 3'))

        # write original image with added contours to disk
        if debug:
            sv = Image.fromarray(img)
            cv2.imshow('captcha_result', img)
            sv.save('test_bounds.png')

        return txt_

    def update_charname(self): # DONE
        x_, y_, w_, h_ = Constants.SCR_CHARNAME
        charname_part = numpy.array(self.snapshot)[y_:y_ + h_, x_:x_ + w_]

        # cv2.imshow("char name",charname_part)

        charname_line = self.txt_from_img(charname_part, (80, 200), (10, 200), yshift=-1, dil_or=(9, 3), fac=2)
        self.char['name'] = charname_line[0]
        if debug:
            print 'CHAR NAME: '
            print charname_line[0]
            print '----------------------------'

    def update_lvl(self): # DONE
        x_, y_, w_, h_ = Constants.SCR_LVL
        lvl_part = numpy.array(self.snapshot)[y_:y_ + h_, x_:x_ + w_]

        # cv2.imshow("lvl",lvl_part)

        lvl_line = self.txt_from_img(lvl_part, (80, 200), (30, 200), yshift=0, dil_or=(9, 3), fac=1, cfg_str= "-c tessedit_char_whitelist=0123456789 --oem 3 --psm 7")
        self.lvl = lvl_line
        if debug:
            print 'LVL: '
            print lvl_line[0]
            print '----------------------------'

    def update_exp(self): # DONE
        x_, y_, w_, h_ = Constants.SCR_EXP
        exp_part = numpy.array(self.snapshot)[y_:y_ + h_, x_:x_ + w_]

        # cv2.imshow("exp",exp_part)

        exp_line = self.txt_from_img(exp_part, (120, 250), (100, 250), yshift=0, dil_or=(9, 3), fac=2, cfg_str= "-c tessedit_char_whitelist=0123456789.,% --oem 3 --psm 7")
        exp_line = float(exp_line[0].strip('%'))
        self.exp = exp_line

        # exp_line = pytesseract.image_to_string(exp_part,config="-c tessedit_char_whitelist=0123456789_., --oem 3 --psm 7")
        if debug:
            print 'EXP: '
            print exp_line
            print '----------------------------'

    def update_adena(self):  # DONE
        x_, y_, w_, h_ = Constants.SCR_ADENA
        adena_part = numpy.array(self.snapshot)[y_:y_ + h_, x_:x_ + w_]

        adena_line = self.txt_from_img(adena_part, (120, 200), (60, 200), yshift=0, dil_or=(9, 3), fac=2, cfg_str="-c tessedit_char_whitelist=0123456789, --oem 3 --psm 7")
        # adena_line = pytesseract.image_to_string(adena_part,config="-c tessedit_char_whitelist=0123456789, --oem 3 --psm 7")
        self.adena = (adena_line)
        if debug:
            print 'ADENA: '
            print adena_line
            print '----------------------------'

    def update_char(self):
        char_ = {'lvl': -1, 'name': "", 'cp': [-1, -1, -1], 'hp': [-1, -1, -1], 'mp': [-1, -1, -1]}

        # GET CROP OF SNAP: ROI
        x_, y_, w_, h_ = Constants.SCR_CHAR
        char_part = numpy.array(self.snapshot)[y_:y_ + h_, x_:x_ + w_]
        x_,y_,w_,h_ = Constants.SCR_CP
        cp_part = numpy.array(self.snapshot)[y_:y_+h_,x_:x_+w_]
        x_, y_, w_, h_ = Constants.SCR_HP
        hp_part = numpy.array(self.snapshot)[y_:y_ + h_, x_:x_ + w_]
        x_, y_, w_, h_ = Constants.SCR_MP
        mp_part = numpy.array(self.snapshot)[y_:y_ + h_, x_:x_ + w_]

        hpline = self.txt_from_img(hp_part, (100, 200), (80, 200), yshift=0, dil_or=(9, 3), fac=2,cfg_str="-c tessedit_char_whitelist=0123456789-|:;/ --oem 3 --psm 7")
        cpline = self.txt_from_img(cp_part, (120, 200), (80, 200), yshift=-5, dil_or=(9, 3), fac=2,cfg_str="-c tessedit_char_whitelist=0123456789-|:;/ --oem 3 --psm 7")
        # hpline = self.txt_from_img(hp_part,(140,200),(10,200),yshift=0,dil_or=(9,3),fac=1)
        mpline = self.txt_from_img(mp_part,(130,200),(80,200),yshift=-2,dil_or=(9,3),fac=2,cfg_str="-c tessedit_char_whitelist=0123456789-|:;/ --oem 3 --psm 7")
        lines = cpline+hpline+mpline
        #lines = self.txt_from_img(char_part,(110,200),(100,200),yshift=-1,dil_or=(9,3),fac=2)
        # print lines
        # GET VALUES
        ind_ = 1
        for l in lines:
            if '/' in l:
                spl_ = l.split('/')
                if ind_ == 3:
                    mp_curr = float(spl_[0])
                    mp_max = float(spl_[1])
                    mp_per = (mp_curr/mp_max)*100

                    char_['mp'] = [int(mp_curr), int(mp_max), mp_per]

                elif ind_ == 2:
                    hp_curr = float(spl_[0])
                    hp_max = float(spl_[1])
                    hp_per = (hp_curr/hp_max)*100

                    char_['hp'] = [int(hp_curr), int(hp_max), hp_per]

                elif ind_ == 1:
                    cp_curr = float(spl_[0])
                    cp_max = float(spl_[1])
                    cp_per = (cp_curr/cp_max)*100

                    char_['cp'] = [int(cp_curr), int(cp_max), cp_per]

                # elif ind_ == 4:
                #     # LVL NAME
                #     spl_ = spl_.split(' ')
                #     lvl_ = int(spl_[0])
                #     name_ = spl_[1]
                #
                #     char_['lvl'] = lvl_
                #     char_['name'] = name_
            elif ';' in l:
                spl_ = l.split(';')
                if ind_ == 3:
                    mp_curr = float(spl_[0])
                    mp_max = float(spl_[1])
                    mp_per = (mp_curr / mp_max) * 100

                    char_['mp'] = [int(mp_curr), int(mp_max), mp_per]

                elif ind_ == 2:
                    hp_curr = float(spl_[0])
                    hp_max = float(spl_[1])
                    hp_per = (hp_curr / hp_max) * 100

                    char_['hp'] = [int(hp_curr), int(hp_max), hp_per]

                elif ind_ == 1:
                    cp_curr = float(spl_[0])
                    cp_max = float(spl_[1])
                    cp_per = (cp_curr / cp_max) * 100

                    char_['cp'] = [int(cp_curr), int(cp_max), cp_per]

                # elif ind_ == 4:
                #     # LVL NAME
                #     spl_ = spl_.split(' ')
                #     lvl_ = int(spl_[0])
                #     name_ = spl_[1]
                #
                #     char_['lvl'] = lvl_
                #     char_['name'] = name_

            ind_ += 1
        # print 'character region: '
        # print lines
        self.char = char_
        if debug:
            print 'CHAR[]'
            print self.char
            print '----------------------------'

    def update_target(self):
        targ_ = {'name': "", 'hp': -1, 'mp': -1}
        # GET CROP OF SNAP: ROI
        x_, y_, w_, h_ = Constants.SCR_TARGET
        snapshot_part = numpy.array(self.snapshot)[y_:y_ + h_, x_:x_ + w_]

        lines = self.txt_from_img(snapshot_part, (140, 200), (10, 200),cfg_str='--oem 3 --psm 7',fac=4)
        if debug:
            print 'target region: '
            print lines
            print '----------------------------'

        if len(lines)>2:
            targ_['name'] = lines[1]
        elif len(lines)==2:
            targ_['name'] = lines[0]
        self.target = targ_

    def get(self):
        self.snapshot = self.img_from_win()