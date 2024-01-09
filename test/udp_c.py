#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.1", 6000)

AAAA = { "a":1}

init_flag = True
while True:
	# data = input("Please input your name: ")
	if init_flag:
		data = {"session_type":"initialize"}
		data = json.dumps(data)
		s.sendto(data.encode(), addr)
		init_flag = False
	response, addr = s.recvfrom(1024)
	print(response.decode())
