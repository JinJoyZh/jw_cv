#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

addr = ("127.0.0.1", 5000)

flight_data = {
					"data_type":"flight_data",
					"source": "rtsp://127.0.0.1:8554/mystream",
					"flight_data":{
						"time":11111,
						"roll":20,
						"pitch":45,
						"yaw":10,
						"camera_latitude":39.9042,
						"camera_longitude":116.4074 ,
						"camera_altitude": 400,
						"focal_length":35
					}
				}
flight_data = json.dumps(flight_data)

counter = 0
while True:
	counter += 1
	#发送一条初始化消息
	if counter == 1:
		data = {"session_type":"initialize"}
		data = json.dumps(data)
		s.sendto(data.encode(), addr)
	#发送飞参
	if counter > 3:
		s.sendto(flight_data.encode(), addr)
	response, addr = s.recvfrom(1024)
	print(response.decode())


	