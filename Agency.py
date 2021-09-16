import ctypes
import json
import msvcrt
import win32gui
import win32api
import os
import numpy
import pickle
import subprocess
import time
import datetime
import sys
import socket
import keyboard
import time
import thread

from classes import UDP
from win32gui import GetForegroundWindow, GetWindowText
from colorama import init, Fore, Back, Style
from termcolor import colored, cprint
from bottle import run, post, request, response


# config
UDPurl = 'localhost'
bindTo = ['Kladblok', 'Lineage II', 'Notepad', 'Windows Command Processor']
debug = False
debuglvl = 0
# init
init()
# Register initialisation
# To keep track of Sat and Agents
Register = []
portRequestTimer = 0
runAgency = True
keyb_timer = 0

frequency = 60  # Hz
period = 1.0/frequency


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def getPortRandom():
    port_ = numpy.random.random_integers(40000,50000,1)[0]
    return port_


def flush_input():
    try:
        import sys, termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except ImportError:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()


def stopAgency():
    print "In a while crocodile!"
    for entry in Register:
        entry['proc'].terminate()
        # entry['sock'].shutdown()
        entry['sock'].close()

    sys.exit()


def terminateAgent(agentName):
    for entry in Register:
        if entry['type'] == 'agent' and entry['name'] == agentName:
            agtProc = entry['proc']
            agtProc.terminate()
            Register.remove(entry)
            print "Withdrawn agent {0} with great succes".format(entry['name'])
    print cprint('agent withdrawal finished. Failed unless informed otherwise.','grey')


def setupSat():
    # sat = subprocess.Popen(['pythonw' 'Satellite.pyw'] + sys.argv[1:], stdout=subprocess.PIPE, shell=True)
    cprint("Setting up tools...",'blue')
    sock_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port_ = getPortRandom()
    newUDP(sock_, UDPurl, port_, 600)
    strport = str(port_)
    sat = subprocess.Popen(["pythonw", "Satellite.pyw", UDPurl, strport])
    Register.append({'type': 'sat', 'proc': sat, 'name': '', 'sock': sock_})
    cprint("Great succes!",'green')
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
        cprint("Assigning agent to box in {0} ... ".format(timer),'yellow')
        timer -= 1
        time.sleep(1)

    wTitle_ = getWinTitle()

    foundIn_ = False
    for bt_ in bindTo:
        if bt_ in wTitle_:
            foundIn_ = True

    if foundIn_:
        sock_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port_ = getPortRandom()
        newUDP(sock_, UDPurl, port_, 600)
        wHandle_ = getWin()
        if debug:
            print wHandle_
        agt = subprocess.Popen(["pythonw", "Agent.pyw", UDPurl, str(port_)])
        Register.append({'wHandle': wHandle_, 'type': 'agent', 'proc': agt, 'name': agentName, 'port': 0, 'sock': sock_})
    else:
        cprint("Failed: Not a supported program ({0})".format(wTitle_),'red')


def getWin():
    # print GetWindowText(GetForegroundWindow())
    fgw = GetForegroundWindow()

    print ("active window: ",  fgw)
    return fgw


def getWinTitle():
    print ("active window title: ", GetWindowText(GetForegroundWindow()))
    return GetWindowText(GetForegroundWindow())


# def checkKeys():
#     if keyboard.is_pressed('s'):
#         print 'f1'
#
#     elif keyboard.is_pressed('esc'):
#         print 'esc'
#
#     elif keyboard.is_pressed('enter'):
#         print 'enter'


def newUDP(sock_, url_, port_, timeout_):
    thread.start_new_thread(listenUDP, (sock_, url_, port_, timeout_))


def listenUDP(sock_, url_, port_, timeout_):
    sock_.bind((url_, port_))
    sock_.settimeout(timeout_)

    timedOut_ = False
    while not timedOut_:
        # print 'receiving...'
        try:
            recv_ = sock_.recvfrom(128 * 1024)
            handlePacket(sock_, recv_)
        except socket.timeout:
            cprint('timed out','red')
            sock_.shutdown()
            sock_.close()
            timedOut_= True


def handlePacket(sock_, pack_):

            # req_obj = json.loads(request.body.read())
            dat_ = pack_[0]
            if dat_ == "resp":
                return
            dec = pickle.loads(dat_)

            # print str(datetime.time.second) + ' ' + str(datetime.time.microsecond)
            if 'Key' in dec:  # and keyb_timer == 0
                src = dec['Window']
                # do something with req_obj
                key = dec['Key']

                # general forward to agent
                for entry_ in Register:
                    if 'wHandle' in entry_:
                        if entry_['wHandle'] == src and entry_['port'] is not 0:
                            sock_.sendto(dat_, ('localhost', entry_['port']))

                # if key == 'Return':
                #     pass
                #     # print Register
                #     # print ' ENTER was pressed'
                #     # print Register
                #
                # elif key == 'S':
                #     assignAgent()
                #     # print ' F1 was pressed'
                #     # assignAgent()
                #
                # elif key == 'Escape':
                #     stopAgency()
                #
                #     # print ' ESCAPE was pressed'
                #     # stopAgency()

            elif 'msgID' in dec:
                if debug:
                    print 'msgID inc'

                if dec['msgID'] == 'agentDebug':
                    print dec['debugMsg']

                elif dec['msgID'] == 'agentFirstSignal':
                    print 'agent first signal'
                    port_ = pack_[1][1]
                    for e_ in Register:
                        if e_['sock'] == sock_:
                            e_['port'] = port_
                    print 'port is ' + str(port_)
                elif dec['msgID'] == 'printmsg':
                    cls()
                    prnt = dec['toprint']
                    if 'subid' in dec:
                        if dec['subid']=='visdat':
                            prnt = prnt['char']
                    print 'received vision update: '
                    print prnt
                    print '----------------------------'


def command():
    cmd_ = str(raw_input("System: "))
    cmd_ = cmd_.split(" ")
    print cmd_
    if cmd_[0] == 'localchat':
        print 'searching for local chat...'

    elif cmd_[0] == 'sysfind':
        print 'searching for [{0}] in system messages...'.format(cmd_[1])

    elif cmd_[0] == 'chatfind':
        print 'searching for [{0}] in chat...'.format(cmd_[1])

    elif cmd_[0] == 'add':
        if len(cmd_) > 1:
            # name is given; dont prompt
            to_add_ = cmd_[1]
        else:
            # prompt for name
            assignAgent()
    elif cmd_[0] == 'exit':
        stopAgency()
    elif cmd_[0] == 'withdraw':
        if len(cmd_) > 1:
            to_withdraw = cmd_[1]
            terminateAgent(to_withdraw)
        else:
            cprint('Provide name of agent to withdraw')
    elif cmd_[0] == 'hide':
        time.sleep(3)
        itemnum = 0
        # win32gui.SendMessage(getWin(), 4104, 10100, 1)
        win32gui.ShowWindow(getWin(), False)
    else:
        if debug:
            print ''

agencyWin = getWin()
setupSat()


# print 'unable to start satellite thread'

secTimer = 0

while runAgency:
    # print 'Agency main loop'

    command()


    # time_before = time.time()
    # do stuff, or wait, or w/e
    # print 'checking keys...'

    # print '</Agency MainLoop>'
    # if keyb_timer > 0:
    #     keyb_time = keyb_timer - period
    #     if keyb_timer < 0:
    #         keyb_timer = 0
    #
    # while (time.time() - time_before) < period:
    #     time.sleep(0.001)  # precision here

    time.sleep(1)


raise SystemExit


