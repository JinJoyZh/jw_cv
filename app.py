import json
import os
import time
import traceback
import cv2
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import socket
from souce_sate import SourceStatesKeeper

from track import run_auto_track
from util.file_helper import get_yaml

import sys
sys.path.insert(1, './alg')
from KCFpy.kcftracker import KCFTracker

SOURCE_STATUS_UNKNOWN   = -1
SOURCE_STATUS_READY     = 0
SOURCE_STATUS_RUNNING   = 1

server = Flask(__name__)

config = get_yaml('config.yaml')

UDP_SERVER_HOST = config['UDP_SERVER_HOST']
UDP_SERVER_PROT = config['UDP_SERVER_PORT']
HTTP_SERVER_PORT = config['HTTP_SERVER_PORT']
MAX_BYTES = config['MAX_BYTES']

thread_pool = ThreadPoolExecutor(max_workers=12)

udp_client_address = None
data_cache = {}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_SERVER_HOST, UDP_SERVER_PROT))

def UDP_start_listening():
    global sock
    print('Listening at', sock.getsockname())
    while True:
        global udp_client_address
        data, udp_client_address = sock.recvfrom(MAX_BYTES) 
        msg = data.decode('ascii')
        msg = json.loads(msg)
        # print(f'receive from client {udp_client_address}, content is: {msg}')
            # {
            #     "rtsp_url":{
            #         {
            #             "flight_data":{}
            #         }
            #     }
            # }
        if 'data_type' in msg:
            source = msg['source']
            if source not in data_cache:
                data_cache[source] = {}
            data_cache[source][msg['data_type']] = msg[msg['data_type']]
            
        if 'session_type' in msg and msg["session_type"] == "initialize":
            res = {'session_type' : 'initialize', 'status':'OK'}
            res = json.dumps(res)
            sock.sendto(res.encode('ascii'), udp_client_address)

@server.route('/jw/alg', methods=['POST'])
def call_algorithm():
    try:
        msg = request.get_json()
        print(msg)
    except Exception as e:
        return jsonify({'error': '请求数据失败'}), 400
    global result
    if msg:
        try:
            global data_cache
            result = exec_task(msg, data_cache)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': '处理数据失败'}), 666
    if not result:
        result = "error"
    return {"output_path": result}

@server.route('/jw/cmd', methods=['POST'])
def call_busyness():
    try:
        msg = request.get_json()
        print(msg)
    except Exception as e:
        return jsonify({'error': '请求数据失败'}), 400
    global result
    if msg:
        try:
            update_source_status(msg)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': '处理数据失败'}), 666
    if not result:
        result = "error"
    return {"output_path": result}

def exec_task(msg, data_cache):
    source_status = update_source_status(msg)
    session_type = msg['session_type']
    if session_type == 'exec_alg' and source_status == SOURCE_STATUS_READY:
        thread_pool.submit(run_auto_track, msg, emit_frame_prediction, data_cache)
    if session_type == "manual_labelling":
        if 'cmd' not in msg:
            return 0
        if msg['cmd'] == "create":
            thread_pool.submit(create_trackers, msg)
        if msg['cmd'] == "delete":
            thread_pool.submit(delete_trackers, msg)
    return 1

def update_source_status(msg):
    status = SOURCE_STATUS_UNKNOWN       #源运行状态 -1:未知 0:待运行 1:正在运行
    if 'source' not in msg:
        return
    states_keeper = SourceStatesKeeper()
    source_state_array = states_keeper.source_state_array
    if 'session_type' in msg:
         #更新算法状态
        if msg['session_type'] == "exec_alg":
            is_new_item = True
            for i in range(0, len(source_state_array)):
                if 'source' not in source_state_array[i]:
                    continue
                if source_state_array[i]['source'] == msg['source']:
                    source_state_array[i] = msg
                    is_new_item = False
                    status = SOURCE_STATUS_RUNNING
                    break
            if is_new_item:
                source_state_array.append(msg)
                status = SOURCE_STATUS_READY
        #更新业务状态
        if msg['session_type'] == "scream_shot":
            for i in range(0, len(source_state_array)):
                if 'source' not in source_state_array[i]:
                    continue
                if source_state_array[i]['source'] == msg['source']:
                    source_state_array[i]["screen_shot"] = 1
                    source_state_array[i]["receive_time_s"] = msg['send_time']
                    break
        if msg['session_type'] == "record_video":
            for i in range(0, len(source_state_array)):
                if 'source' not in source_state_array[i]:
                    continue
                if source_state_array[i]['source'] == msg['source']:
                    source_state_array[i]["save_videos"] = msg['save_videos']
                    source_state_array[i]["receive_time_v"] = msg['send_time']
                    break
        if msg['session_type'] == "show_box":
            for i in range(0, len(source_state_array)):
                if 'source' not in source_state_array[i]:
                    continue
                if source_state_array[i]['source'] == msg['source']:
                    source_state_array[i]["draw_box"] = msg['draw_box']
                    break
        if msg['session_type'] == "adjust_boudingbox":
            for i in range(0, len(source_state_array)):
                if 'source' not in source_state_array[i]:
                    continue
                if source_state_array[i]['source'] == msg['source']:
                    update_boundingboxes(msg, source_state_array[i])
                    break
    return status

# 手动标签, 更新追踪器
def create_trackers(msg):
    if 'session_type' not in msg or msg['session_type'] != "manual_labelling":
        return
    states_keeper = SourceStatesKeeper()
    source_state_array = states_keeper.source_state_array
    source_index = -1
    for i in range(0, len(source_state_array)):
        if 'source' not in source_state_array[i]:
            continue
        if source_state_array[i]['source'] == msg['source']:
            source_index = i
            break
    if source_index == -1:
        return
    if 'width' not in source_state_array[source_index] or "height" not in source_state_array[source_index]:
        return
    config = get_yaml('config.yaml')
    screen_shot_dir = config['SHARED_SCREEN_SHOT_DIR']
    time_zone = str(msg['shot_time'])
    img_file_path = screen_shot_dir + time_zone + ".jpg"
    #在3秒内持续检查是否有相关图片生成
    counter = 0
    gap = 0.05
    limit = int(3.0 / gap)
    while True:
        if not os.path.exists(img_file_path):
            if counter < limit:
                counter += 1
                continue
        break
    if counter == limit:
        print(f"Error! manual_labelling {img_file_path} not exist")
        return
    frame = cv2.imread(img_file_path)
    manual_targets = msg['targets']
    #初始化目标跟踪器
    if 'trackers' not in source_state_array[source_index]:
        source_state_array[source_index]['trackers'] = {}
    #构造每个目标的跟踪器
    real_width = source_state_array[source_index]['width']
    real_height = source_state_array[source_index]['height']
    x_zoom = real_width / msg['display_width']
    y_zoom = real_height / msg['display_heigh']
    for j in range(0, len(manual_targets)):
        tracker = KCFTracker(True, True, True)  # hog, fixed_window, multiscale
        tar = manual_targets[j]
        x = tar['x1'] * x_zoom
        y = tar['y1'] * y_zoom
        w = (tar['x2'] - tar['x1']) * x_zoom
        h = (tar['y2'] - tar['y1']) * y_zoom
        tracker.init([int(x), int(y), int(w), int(h)], frame)
        label = "NONE"
        if "label" in tar:
            label = tar["label"]
        id = id(tracker)
        id = str(id)
        source_state_array[source_index]['trackers'][id] = {"tracker": tracker, "label": label}

def delete_trackers(msg):
    if 'session_type' not in msg or msg['session_type'] != "manual_labelling":
        return
    states_keeper = SourceStatesKeeper()
    source_state_array = states_keeper.source_state_array
    source_index = -1
    for i in range(0, len(source_state_array)):
        if 'source' not in source_state_array[i]:
            continue
        if source_state_array[i]['source'] == msg['source']:
            source_index = i
            break
    if source_index == -1:
        return
    if 'trackers' not in source_state_array[source_index]:
        return
    delete_targets = msg['targets']
    for target in delete_targets:
        source_state_array[source_index]['trackers'].pop(target, None)


# 调整外框大小
def update_boundingboxes(msg, source_state):
    adjustment_array = []
    context = msg['context']
    for item in context:
        adjust = {}
        adjust['id'] = item['id']
        before = item['before']
        after = item['after']
        old_width = abs(before['x1'] - before['x2'])
        new_width = abs(after['x1'] - after['x2'])
        scale_w = float(new_width) / float(old_width)
        old_height = abs(before['y1'] - before['y2'])
        new_height = abs(after['y1'] - after['y2'])
        scale_h = float(new_height) / float(old_height)
        o_center_x = (before['x1'] + before['x2']) / 2
        o_center_y = (before['y1'] + before['y2']) / 2
        n_center_x = (after['x1'] + after['x2']) / 2
        n_center_y = (after['y1'] + after['y2']) / 2
        shift_x = n_center_x - o_center_x
        shift_y = n_center_y - o_center_y
        adjust['scale_w'] = scale_w
        adjust['scale_h'] = scale_h
        adjust['shift_x'] = shift_x
        adjust['shift_y'] = shift_y
        adjustment_array.append(adjust)
    source_state['adjust_boudingbox'] = adjustment_array

def emit_frame_prediction(prediction):
    print(f"emit_frame_prediction {prediction}")
    if not udp_client_address:
        print('WARNING! UDP_CLIENT_ADDRESS == NONE')
        return
    global sock
    sock.sendto(prediction.encode('ascii'), udp_client_address)
    print(f" [x] Sent :{prediction}")

if __name__ == '__main__':
    thread_pool.submit(UDP_start_listening)
    server.run(host='0.0.0.0', port=HTTP_SERVER_PORT)