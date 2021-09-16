import os
import win32gui

import pyHook

import pythoncom

from win32gui import GetWindowText, GetForegroundWindow
from bottle import run, post, request, response

# pId = os.getpid()

# LVM_DELETEITEM = 4104
# win32gui.SendMessage(listview, LVM_DELETEITEM, pId, 0)

def onKeyboardEvent(event):

    # no use printing in pyw
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

    evt = [event.MessageName, event.Message, ]

    # return True to pass the event to other handlers
    return True


# create a hook manager
hm = pyHook.HookManager()

# watch for all mouse events
hm.KeyDown = onKeyboardEvent
# hm.KeyUp = onKeyboardEvent

# set the hook
hm.HookKeyboard()

# start listening
pythoncom.PumpMessages()



