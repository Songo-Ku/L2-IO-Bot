# class UDP:


def send(sock, addr, port, msg):
    print "trying to send..."
    try:
        sent = sock.sendto(msg, (addr, port))
    except:
        print "failed"
        sent = None

    return sent


def recv(sock, port):
    print "trying to receive..."

    try:
        data, server = sock.recvfrom(port)
    except:
        print "failed"
        data = None

    return data
