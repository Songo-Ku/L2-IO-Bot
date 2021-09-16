import ctypes
import json
import msvcrt
import numpy
import pickle
import subprocess
import time
import datetime
import sys
import socket
import keyboard

from classes import UDP
from win32gui import GetForegroundWindow, GetWindowText
from bottle import run, post, request, response


# config
runAgency = True

UDPAddr = 'localhost'
UDPPort = 41330

# init
# Register initialisation
# To keep track of Sat and Agents
Register = []


def flush_input():
    try:
        import sys, termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except ImportError:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()


def getSock():
    return sock


def stopAgency():
    print "In a while crocodile!"
    for entry in Register:
        entry['proc'].terminate()

    sock.close()
    sys.exit()


def terminateAgent(agentName):
    for entry in Register:
        if entry['type'] == 'agent' and entry['name'] == agentName:
            agtProc = entry['proc']
            agtProc.terminate()
            Register.remove(entry)
            print "Withdrawn agent {0} with great succes".format(entry['name'])


def setupSat():
    # sat = subprocess.Popen(['pythonw' 'Satellite.pyw'] + sys.argv[1:], stdout=subprocess.PIPE, shell=True)
    print "Setting up tools..."
    sat = subprocess.Popen(["pythonw", "Satellite.pyw"])
    Register.append({'type': 'sat', 'proc': sat, 'name': ''})
    print "Great succes!"
    return sat


def assignAgent():

    # create Agent process
    print "Provide agent name and press ENTER. Then, after 5 seconds " \
          "the agent will automatically bind to the foreground window. " \
          "Make sure to be in that window!"
    # print "Agent name: "

    flush_input()
    agentName = raw_input("Agent name: ")

    timer = 5
    while timer > 0:
        print "Assigning agent to box in {0} ... ".format(timer)
        timer -= 1
        time.sleep(1)

    if "Kladblok" in getWinTitle():
        agt = subprocess.Popen(["pythonw", "Agent.pyw", agentName])
        Register.append({'type': 'agent', 'proc': agt, 'name': agentName})
        waitresp_ = True
        while waitresp_:
            (dat_, adr_) = sock.recvfrom(128 * 1024)
            if 'port' in dat_
    else:
        print "Failed: Not a 'Kladblok' window ({0})".format(getWinTitle())


def getWin():
    # print GetWindowText(GetForegroundWindow())
    fgw = GetForegroundWindow()

    print ("active window: ",  fgw)
    return fgw


def getWinTitle():
    print ("active window title: ", GetWindowText(GetForegroundWindow()))
    return GetWindowText(GetForegroundWindow())


def checkKeys():
    if keyboard.is_pressed('f1'):
        print 'f1'
        assignAgent()
    elif keyboard.is_pressed('esc'):
        print 'esc'
        stopAgency()
    elif keyboard.is_pressed('enter'):
        print 'enter'
        print Register


def handlePacket(dataIN):
    print 'Incoming key report:'
    # req_obj = json.loads(request.body.read())
    dec = pickle.loads(dataIN)
    # print str(datetime.time.second) + ' ' + str(datetime.time.microsecond)
    # get source of keypress

    if 'Key' in dec:
        src = dec['Window']
        # do something with req_obj
        key = dec['Key']

        # general forward to agent
        for entry_ in Register:
            if entry_['proc'] == src and entry_['port'] is not 0:
                sock.sendto()

        # ageny specific
        if key == 'Return':
            pass
            # print ' ENTER was pressed'
            # print Register

        elif key == 'F1':
            pass
            # print ' F1 was pressed'
            # assignAgent()

        elif key == 'Escape':
            pass

            # print ' ESCAPE was pressed'
            # stopAgency()

        elif key == 'H':
            # print ' H was pressed'
            pass
    elif 'msgID' in dec:
        if dec['msgID'] == 'agentFirstSignal':



    return 'All done'


setupSat()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDPAddr, UDPPort))
# start key listener
print 'post-bind'

secTimer = 0
while runAgency:
    # do stuff, or wait, or w/e
    (dat, adr) = sock.recvfrom(128*1024)
    handlePacket(dat)
    print 'mloop'
    checkKeys()
    # time.sleep(1)

raise SystemExit


