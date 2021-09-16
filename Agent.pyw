import ctypes
import msvcrt
import pickle
import socket
import subprocess
import thread
import time
import win32api
import win32gui
import sys

import autoit
import numpy
import win32con

from classes import Vision
from multiprocessing import  Pool


# settings
debug = False


def multi_run_wrapper(args):
    return doVision(*args)


def doVision(update_code):
    re = -1
    if update_code == 0:
        re = vision.update_charname()
    elif update_code == 1:
        re = vision.update_lvl()
    elif update_code == 2:
        re = vision.update_exp()
    elif update_code == 3:
        re = vision.update_adena()
    elif update_code == 4:
        re = vision.update_char()

    return re


def getPortRandom():
    port_ = numpy.random.random_integers(40000,50000,1)[0]
    return port_


def newUDP(sock_, url_, port_, timeout_):
    thread.start_new_thread(listenUDP, (sock_, url_, port_, timeout_))


def handleTimerRun():
    timer_ = 100


def listenUDP(sock_, url_, port_, timeout_):
    sock_.bind((url_, port_))
    sock_.settimeout(timeout_)

    timedOut_ = False
    while not timedOut_:
        if debug:
            print 'agent is receiving...'

        try:
            recv_ = sock_.recvfrom(128 * 1024)
            agentHandlePacket(recv_)
        except socket.timeout:
            print "agent's socket timed out"
            sock_.shutdown()
            sock_.close()
            timedOut_ = True


def agentHandlePacket(pack_):
    # req_obj = json.loads(request.body.read())
    dat_ = pack_[0]
    dec_ = pickle.loads(dat_)

    if 'Key' in dec_:
        key_ = dec_['Key']
        if debug:
            print '<<<<<< AAAGEENTTT REECEEEIIVVEEEDDD KEEEYYYY >>>>>>>'
            print '----------------------------'
        sock.sendto("resp", (agencyURL, agencyPORT))
        if dec_['Key'] == 'H':
            # send_input_hax(boxHandle, win32con.VK_ESCAPE) WORKS
            # time.sleep(0.1)
            send_msg_hax(boxHandle,"ello!")
            # win32gui.SendMessage(boxHandle, win32con.WM_CHAR, ord('b'), 0) WORKS
        elif dec_['Key'] == 'Return':
            sendpack_ = {'msgID': 'printmsg', 'toprint': ''}
            if vision.isSaidLocal():
                sendpack_['toprint'] = "__________+++++++++++++++++_____________"
                if debug:
                    print '__________LOCAL CHAT FOUND ___________'
                    print '----------------------------'
            else:
                sendpack_['toprint'] = "__________-----------------_____________"
                if debug:
                    print 'NO LOCAL CHAT FOUND____________________'
                    print '----------------------------'
            sock.sendto(pickle.dumps(sendpack_),(agencyURL,agencyPORT))
    # print str(datetime.time.second) + ' ' + str(datetime.time.microsecond)
    if 'msgID' in dec_:
        if dec_['msgID' == 'agentFirstSignal']:
            print 'agent first signal inc'
            port_ = pack_[1][1]
            print 'port is ' + str(port_)


def send_input_key(hwnd, key):
    win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, key, 0)
    time.sleep(0.05)
    win32api.SendMessage(hwnd, win32con.WM_KEYUP, key, 0)
    # win32gui.SendMessage()


def send_input_msg(hwnd, msg):
    for c in msg:
        if c == "\n":
            win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
            time.sleep(0.05)
            win32api.SendMessage(hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
        else:
            win32api.SendMessage(hwnd, win32con.WM_CHAR, ord(c), 0)
        time.sleep(0.15)


def setTitle(title):
    # set Agent's console title
    ctypes.windll.kernel32.SetConsoleTitleA("l2Agent - " + title)


def getWin():
    # print GetWindowText(GetForegroundWindow())
    fgw = win32gui.GetForegroundWindow()

    print ("active window: ",  fgw)
    return fgw


def send_input_hax(hwnd, key_):
    win32gui.SendMessage(hwnd, win32con.WM_KEYDOWN, key_, 0)
    time.sleep(0.01)
    win32gui.SendMessage(hwnd, win32con.WM_KEYUP, key_, 0)


def send_msg_hax(hwnd, msg_):
    for c in msg_:
        if c == "\n":
            win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
            time.sleep(0.01)
            win32api.SendMessage(hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
        else:
            win32api.SendMessage(hwnd, win32con.WM_CHAR, ord(c), 0)
        time.sleep(0.1)


def send_input_hax_send(hwnd, msg):
    for c in msg:
        if c == "\n":
            win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
            win32api.SendMessage(hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
        else:
            win32api.SendMessage(hwnd, win32con.WM_CHAR, ord(c), 0)


def send_input_hax_post(hwnd, msg):
    for c in msg:
        if c == "\n":
            win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
            win32api.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
        else:
            win32api.PostMessage(hwnd, win32con.WM_CHAR, ord(c), 0)


if __name__ == "__main__":
    # set console title to character name
    # charName = sys.argv[2]
    # setTitle(charName)

    # boxHandle = getWin()
    boxHandle = autoit.win_get_handle("")
    vision = Vision.Vision(boxHandle)

    if debug:
        print boxHandle
        print autoit.win_get_title_by_handle(boxHandle)
    agencyURL = sys.argv[1]
    agencyPORT = int(sys.argv[2])
    if debug:
        print "agent: agency url is: " + agencyURL
        print "agent thinks agency has port: " + str(agencyPORT)

    # MAKE SOCKET -> SEND MESSAGE TO AGENCY -> agency links port to entry
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    agentPort = getPortRandom()
    if debug:
        print 'agent starts udp...'
    newUDP(sock,agencyURL,agentPort,600)
    if debug:
        print 'post- udp start'

    first_signal_r = {'msgID': 'agentFirstSignal', 'port': agentPort}
    first_signal = pickle.dumps(first_signal_r)
    if debug:
        print 'agent: sending first...'
    sock.sendto(first_signal, (agencyURL, agencyPORT))
    if debug:
        print 'sent!'

    test_signal = {'msgID': 'agentDebug', 'debugMsg': 'agent his box handle is {0}'.format(boxHandle)}

    frequency = 5  # Hz
    period = 1.0/frequency

    # pool = Pool(4)

    # first box update is full
    vision.update_full()

    # main loop
    while True:
        time_before = time.time()
        # AI
        if debug:
            print 'agent loop'

        vision.update_short()
        # print vision.get_char_mp()

        # send_input_hax(boxHandle, win32con.VK_ESCAPE)
        # results = pool.map(multi_run_wrapper, [0, 1, 2, 3, 4])
        # name_ = pool.apply_async(doVision, [0,vision])
        # lvl_ = pool.apply_async(doVision, [1,vision])
        # exp_ = pool.apply_async(doVision, [2,vision])
        # adn_ = pool.apply_async(doVision, [3,vision])
        # chr_ = pool.apply_async(doVision, [4,vision])
        #a=pool.map_async(doVision,[0,1,2,3,4])
        # print a.get()
        # for info in vision.char:
        #     print vision.char[info]

        # send_msg_hax(boxHandle,".")

        # sock.sendto(pickle.dumps({'msgID': 'printmsg',
        #                          'toprint': [vision.char, vision.target]}),
        #            (agencyURL, agencyPORT))

        # END OF AGENT LOOP: sleep remaining time
        dts_ = time.time() - time_before
        while (time.time() - time_before) < period:
            time.sleep(period - dts_)
