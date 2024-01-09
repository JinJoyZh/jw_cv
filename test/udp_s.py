#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor
import socket
import time

thread_pool = ThreadPoolExecutor(max_workers=12)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 6001))
print("UDP bound on port 6001...")

UDP_CLIENT_ADDRESS = None

def UDP_start_listening():
    global sock
    print('Listening at', sock.getsockname())
    while True:
        global UDP_CLIENT_ADDRESS
        data, UDP_CLIENT_ADDRESS = sock.recvfrom(1024) 
        msg = data.decode('ascii')
        print('The client at {} says {!r}'.format(UDP_CLIENT_ADDRESS, msg))


# while True:
# 	data, addr = s.recvfrom(1024)
# 	print("Receive from %s:%s" % addr)
# 	if data == b"exit":
# 		s.sendto(b"Good bye!\n", addr)
# 		continue
# 	s.sendto(b"Hello %s!\n" % data, addr)

if __name__ == '__main__':
    thread_pool.submit(UDP_start_listening)
    i = 1
    for i in range(10000):
        print(f'round {i}')
        if UDP_CLIENT_ADDRESS:
            print(f'UDP_CLIENT_ADDRESS {UDP_CLIENT_ADDRESS}')
            sock.sendto(b"Hello aa \n", UDP_CLIENT_ADDRESS)
        time.sleep(2)