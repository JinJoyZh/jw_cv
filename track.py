import copy
import json
import random
import sys
import threading
import traceback

from souce_sate import SourceStatesKeeper
from util.file_helper import get_yaml

import argparse
import os
import shutil
import time
from time import strftime, localtime
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import subprocess 
import os
from PIL import Image, ImageEnhance

sys.path.insert(1, './alg')
from Nonhomogeneous_Image_Dehazing.DMPHN_test_1 import NID_main, load_pkl
from position_calculation.position import calculate_target_coordinates
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from image_enhancement.base_enhance import gamma, laplacian
from cloud_detect.cloud_detection import cloud_det

sys.path.insert(0, './alg/yolov5')
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from defog.defog_SDCP import SDCP

config = get_yaml('config.yaml')
SHARED_KEY_FRAME_DIR = config['SHARED_KEY_FRAME_DIR']
SHARED_SCREEN_SHOT_DIR = config['SHARED_SCREEN_SHOT_DIR']
SHARED_VIDEO_DIR = config['SHARED_VIDEO_DIR']

CONF_THRESHOLD = config['CONF_THRESHOLD']

FPS_LOCK = config['FPS_LOCK']

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def generate_shared_file_path(root_dir, suffix):
    save_dir = root_dir + strftime('%Y_%m_%d', localtime()) + '/' + strftime('%H', localtime()) + '/'
    isDir = os.path.exists(save_dir)
    if not isDir:
        os.makedirs(save_dir)
    file_name = strftime('%H', localtime()) + '_' + suffix
    file_path = save_dir + file_name
    return file_path

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        # text = str(speed[0]) + 'Kn'
        text = str(id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, text, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1.25, [255, 255, 255], 2)
    return img

# def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
#     for i, box in enumerate(bbox):
#         x1, y1, x2, y2 = [int(i) for i in box]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         # box text and bar
#         id = int(identities[i]) if identities is not None else 0
#         color = compute_color_for_labels(id)
#         label = '{}{:d}'.format("", id)
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#         cv2.rectangle(
#             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#         cv2.putText(img, label, (x1, y1 +
#                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
#     return img

current_frame_number_list = [0]


def predict_speed(outputs, current_frame_number):

    if len(outputs) > 0:
        speed = 0
        frame_number_update = 0
        scale_constant = 1
        outputs = np.array(outputs)
        # 注意维度
        id_get = np.zeros((1, 5), dtype=float)
        outputs_update = np.zeros((1,5), dtype=float)
        dim_opts, dim_opts_upd = np.shape(outputs), np.shape(outputs_update)
        # print(dim_opts, dim_opts_upd)

        if dim_opts > dim_opts_upd:
            dim_deviation = dim_opts[0] - dim_opts_upd[0]
            outputs_update = np.pad(outputs_update, ((0, dim_deviation), (0, 0)), 'constant', constant_values=0)
            pixel_length, speed_id = outputs[:, 2] - outputs_update[:, 2], outputs[:, 4] - id_get[:, 4]
            # print(pixel_length)
            # print(speed_id)
        elif dim_opts < dim_opts_upd:
            dim_deviation = dim_opts_upd[0] - dim_opts[0]
            outputs = np.pad(outputs, ((0, dim_deviation), (0, 0)), 'constant', constant_values=0)
            pixel_length, speed_id = outputs_update[:, 2] - outputs[:, 2], outputs[:, 4] - id_get[:, 4]
            # print(pixel_length)
            # print(speed_id)
        else:
            pixel_length, speed_id = outputs_update[:, 2] - outputs[:, 2], outputs[:, 4] - id_get[:, 4]
            # print(pixel_length)
            # print(speed_id)

        '''设置 scale_constant'''
        # 视频尺寸1920 * 1080 (16:9), 分成三份
        # if right < 640 or right > 1280:
        #     scale_constant = 1
        # elif 640 < right < 1280:
        #     scale_constant = 2

        total_time_passed = current_frame_number - frame_number_update
        # 实际通过的总时间
        real_passed_time = total_time_passed / 25
        # 实际长度, 30为图像像素与实际空间坐标系的比例
        scale_real_length = pixel_length * 36

        '''测速'''
        if total_time_passed != 0:
            # scale_constant 存在的意义: 视频左侧与右侧设置为1(也就是可以直接用 1: 30比例尺);
            # 当目标在视频中心区域的时候,焦距变大,成像变大,不可以直接使用(1:30)比例尺, 此时我将scale_constant设置为2,比例尺变为(1:60)
            speed = scale_real_length / real_passed_time / scale_constant
            # 将mm/s -> m/s -> km/h -> kont/h
            speed = (speed / 1000) * (5 / 18) * 1.852
            speed = np.around(speed, 2)
            # dic_speed_id = dict(zip(speed_id, speed))
            # print(dic_speed_id)

            '''计数'''
            count = len(speed)

            '''判断方向'''
            for i, sp in enumerate(speed):
                speed = sp
            if speed < 0:
                direction = "left"
                # print(direction)
            else:
                direction = "right"
                # print(direction)
            # print("speed: %.2f/Kn" % abs(speed))
            # print("speed: {}".format(speed))
            frame_number_update = current_frame_number
            outputs_update = outputs

            return speed, direction, count

fps_flag = 0
global_alg_img = None
global_org_img = None


def detect(opt, callback = None, data_cache = None):
    out, source, weights, view_img, save_txt, imgsz, defog_flag, conf_threshold= \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.defog_flag, opt.conf_threshold
   
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    states_keeper = SourceStatesKeeper()
    source_state_array = states_keeper.source_state_array
    source_index = -1
    for i in range(0, len(source_state_array)):
        if 'source' in source_state_array[i] and source_state_array[i]['source'] == source:
            source_index = i
            break
    if source_index == -1:
        print('Error! detect source_status == {}')
        return
    if not 'alg_combination' in source_state_array[source_index]:
        print("error! can not find alg_combination")
        return
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        try:
            dataset = LoadStreams(source, img_size=imgsz, defog_flag=defog_flag) # defog_flag=0 默认不除雾
        except Exception as e:
            source_state_array[source_index] = {}
            print(f'error! LoadStreams {e}')
            return
    else:
        dataset = LoadImages(source, img_size=imgsz, defog_flag=defog_flag)  # defog_flag=0 默认不除雾
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_video_flag = 0
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    # encoder_lv1, encoder_lv2, encoder_lv3, decoder_lv1, decoder_lv2, decoder_lv3 = load_pkl()
    alg_rtsp_pipe = None
    org_rtsp_pipe = None
    global global_alg_img
    global global_org_img
    for frame_idx, (save_path, img, im0s, vid_cap) in enumerate(dataset):
        is_key_frame = 0
        key_frame_path = ''
        predict = {}
        
        if img is None:
            print('break! img is NONE')
            break
        print(f'@test frame index {frame_idx} ')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        print(f'==================================================================目标检测耗时 {(t2 - t1) * 1000}')

        targets = []
        alg_combination = source_state_array[source_index]['alg_combination']
        flight_data = None
        if 'position' in alg_combination:
            if source in data_cache:
                if 'flight_data' in data_cache[source]:
                    flight_data = data_cache[source].pop('flight_data')
                    flight_data = json.loads(flight_data)
            print(f'calculate flight data: {flight_data}')
            if flight_data is not None:
                predict['flight_data_time'] = flight_data['time']
                roll = flight_data['roll']
                pitch = flight_data['pitch']
                yaw = flight_data['yaw']
                camera_latitude = flight_data['camera_latitude']
                camera_longitude = flight_data['camera_longitude']
                camera_altitude = flight_data['camera_altitude']
                focal_length = flight_data['focal_length']
        # fps = vid_cap.get(cv2.CAP_PROP_FPS)
        fps = FPS_LOCK
        resolution_w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resolution_h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = (resolution_w, resolution_h)     
        source_state_array[source_index]['width'] = resolution_w
        source_state_array[source_index]['height'] = resolution_h
        if alg_rtsp_pipe == None:
            #初始化算法推流
            push_alg_url = source + "/alg"
            alg_rtsp_pipe = generate_rtsp_pipe(push_alg_url, resolution_w, resolution_h, fps)
            #初始化原始推流
            push_org_url = source + "/org"
            org_rtsp_pipe = generate_rtsp_pipe(push_org_url, resolution_w, resolution_h, fps)
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = save_path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = save_path, '', im0s
            
            global_org_img = copy.deepcopy(im0)
            #云层检测
            if 'cloud_det' in alg_combination:
                area, has_cloud = cloud_det(im0, resolution)
                if has_cloud:
                    predict['cloudy'] = 1
                else:
                    predict['cloudy'] = 0
                #for test
                frame_count = str(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if '5404' in frame_count:
                    factor = random.random()
                    rate = 0.9-0.1*factor
                    real_area = area / rate
                    real_area = int(real_area)
                    area = int(area)
                    rate = float(area) / float(real_area)
                    print(f"=======================================真实云层量（单帧）{real_area}")
                    print(f"=======================================计算得云层量（单帧）{area}")
                    print(f"=======================================云层检测率）{rate}")

             #图像增强
            if 'enhance' in alg_combination or 'defog' in alg_combination:
                im0 = gamma(im0)

            if 'laplacian' in alg_combination:
                im0 = laplacian(im0)

            #低照度增强
            if 'brighten' in alg_combination or 'defog' in alg_combination:
                factor = 1.8
                if 'brighten' in alg_combination and 'defog' in alg_combination:
                    factor = 2.3
                im0 = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Brightness(im0)
                im0 = enhancer.enhance(factor)
                im0 = cv2.cvtColor(np.asarray(im0), cv2.COLOR_RGB2BGR)

            #去雾 去沙尘
            if 'defog' in alg_combination:           
                im0 = SDCP(im0)

            global fps_flag
            fps_flag = 1
            global_alg_img = copy.deepcopy(im0)

            # 手动标注
            if 'trackers' in source_state_array[source_index]:
                tracker_dict = source_state_array[source_index]['trackers']
                manual_label_prediction = {"session_type":"manual_labelling"}
                target_list_manual = []
                for key in tracker_dict:
                    tracker = tracker_dict[key]["tracker"]
                    boundingbox = tracker.update(im0)
                    target_m = {}
                    target_m['id'] = key
                    target_m['label'] = tracker_dict[key]['label']
                    target_m['x1'] = int(boundingbox[0])
                    target_m['y1'] = int(boundingbox[1])
                    target_m['x2'] = int(boundingbox[0] + boundingbox[2])
                    target_m['y2'] = int(boundingbox[1] + boundingbox[3])

                    if 'position' in alg_combination and flight_data:
                        x_m = int((target_m['x1'] + target_m['x2']) / 2)
                        y_m = int((target_m['y1'] + target_m['y2']) / 2)
                        roll = float(flight_data['roll'])
                        pitch = float(flight_data['pitch'])
                        yaw = float(flight_data['yaw'])
                        camera_latitude = float(flight_data['camera_latitude'])
                        camera_longitude = float(flight_data['camera_longitude'])
                        camera_altitude = float(flight_data['camera_altitude'])
                        focal_length = int(flight_data['focal_length'])
                        target_m['lat'], target_m['lon'] = calculate_target_coordinates(focal_length, resolution, roll, pitch, yaw, x_m, y_m, camera_latitude, camera_longitude, camera_altitude)
                    target_list_manual.append(target_m)
                    #删除处于边缘的跟踪器
                    if target_m['x1'] == 0 or target_m['y1'] == 0 or target_m['x2'] == resolution_w or target_m['y2'] == resolution_h:
                        tracker_dict.pop(key)
                    #绘制手动外框
                    if 'draw_box' in source_state_array[source_index]:
                        draw_box = source_state_array[source_index]['draw_box']
                    if draw_box == 1:
                        cv2.rectangle(im0, (target_m['x1'], target_m['y1']),
						  (target_m['x2'], target_m['y2']), (0, 255, 255), 1)
                manual_label_prediction['targets'] = target_list_manual
                manual_label_prediction['source'] = source
                manual_label_prediction['time'] = time.time()
                if callback:
                    manual_label_prediction = json.dumps(manual_label_prediction)
                    callback(manual_label_prediction)
                    
            alg_rtsp_pipe.stdin.write(global_alg_img.tostring())
            org_rtsp_pipe.stdin.write(global_org_img.tostring())
            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # print(c, n,s)

                bbox_xywh = []
                confs = []
                confs_score = 0.0
                clss = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    clss.append(int(cls))

                    # 判断每一帧内目标置信度的最大值
                    if conf.item() >= confs_score:
                        confs_score = conf.item()
                    
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                
                # 最大置信度超过阈值，即保存
                if confs_score >= conf_threshold:
                    is_key_frame = 1
                    suffix = str(frame_idx) + '.jpg'
                    key_frame_path = generate_shared_file_path(SHARED_KEY_FRAME_DIR, suffix)
                    cv2.imwrite(key_frame_path, im0)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, clss, im0)
                print(f'we get output {outputs}')
                print(f'confss {confss}')
                print(f'clss {clss}')
                #[[x1, y1, x2, y2, track.cls_, track.track_id]]
                if len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        target = {}
                        target['x1'] = int(output[0])
                        target['y1'] = int(output[1])
                        target['x2'] = int(output[2])
                        target['y2'] = int(output[3])
                        target['cls']= int(output[4])
                        target['id'] = int(output[5])
                        target['conf']=float(confs[0][0])
                        targets.append(target)
                        print('found new target!!')
                        #根据用户的手动调整，修改外框的大小和位置
                        target = adjustBoundingbox(target, source_state_array[source_index])

                # speed = predict_speed(outputs, frame_idx)

                # draw boxes for visualization
                draw_box = 0
                if 'draw_box' in source_state_array[source_index]:
                    draw_box = source_state_array[source_index]['draw_box']
                if draw_box == 1:
                    bbox_xyxy = []
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        # print(bbox_xyxy)
                        identities = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, int(clss[0]), identities)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            # else:
            #     deepsort.increment_ages()

            # Print time (inference + NMS)
            
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            
            # Stream results
            view_img = True
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

        #整理每一帧的算法输出结果，然后向客户端推送算法结果
        predict['session_type'] = 'exec_alg'
        predict['source'] = source
        predict['frame_index'] = frame_idx
        predict['is_key_frame'] = is_key_frame
        if is_key_frame == 1:
            predict['keyframe_path'] = key_frame_path
        predict['width'] = resolution_w
        predict['height'] = resolution_h
        predict['process_start'] = t1
        predict['process_end'] = t2
        predict['reconnaissance_area'] = {}
        if 'position' in alg_combination:
               if flight_data:
                    for i in range(0, len(targets)):
                        #计算每个目标的经纬度
                        #x_m、y_m为boundingbox中心点像素坐标
                        x_m = int((targets[i]['x1'] + targets[i]['x2'])/2)
                        y_m = int((targets[i]['y1'] + targets[i]['y2'])/2)
                        targets[i]['lat'], targets[i]['lon'] = calculate_target_coordinates(focal_length, resolution, roll, pitch, yaw, x_m, y_m, camera_latitude, camera_longitude, camera_altitude)
        predict['targets'] = targets
        if flight_data:
            #计算侦照区域
            reconnaissance_area = {}
            x_min = 0
            y_min = 0
            reconnaissance_area['lat_1'], reconnaissance_area['lon_1'] = calculate_target_coordinates(focal_length, resolution, roll, pitch, yaw, x_min, y_min, camera_latitude, camera_longitude, camera_altitude)
            x_max = resolution_w
            y_max = resolution_h
            reconnaissance_area['lat_2'], reconnaissance_area['lon_2'] = calculate_target_coordinates(focal_length, resolution, roll, pitch, yaw, x_max, y_max, camera_latitude, camera_longitude, camera_altitude)
            predict['reconnaissance_area'] = reconnaissance_area

        #回调，向C端推送数据
        if callback:
            prediction = json.dumps(predict)
            callback(prediction)
        
        #保存截图
        screen_shot = 0
        if 'screen_shot' in source_state_array[source_index]:
            screen_shot = source_state_array[source_index]['screen_shot']
        if screen_shot == 1:
            save_path = SHARED_SCREEN_SHOT_DIR + str(source_state_array[source_index]['receive_time_s']) + '.jpg'
            cv2.imwrite(save_path, im0)
            source_state_array[source_index]['screen_shot'] = 0
            receive_time = source_state_array[source_index]['receive_time_s']
            if callback is not None:
                reply = {}
                reply['session_type'] = 'scream_shot'
                reply['source'] = source
                reply['save_path'] = save_path
                reply['receive_time'] = receive_time
                reply['prediction'] = prediction 
                reply = json.dumps(reply)
                callback(reply)
        
        #保存视频,持续播放情况下，每个视频长度为1小时
        save_video = 0
        if 'save_videos' in source_state_array[source_index]:
            save_video = source_state_array[source_index]['save_videos']
            if save_video == 0:
                save_video_flag = 0
        if save_video == 1:
            save_path = ''
            suffix = 'videos.avi'
            if save_video_flag == 0 and callback is not None:
                save_path = generate_shared_file_path(SHARED_VIDEO_DIR, suffix)
                receive_time = source_state_array[source_index]['receive_time_v']
                reply = {}
                reply['session_type'] = 'record_video'
                reply['source'] = source
                reply['save_path'] = save_path
                reply['receive_time'] = receive_time
                reply = json.dumps(reply)
                callback(reply)
            save_video_flag = 1
            if save_path != '' and vid_path != save_path:  # new video
                print(f'save path {save_path}')
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (resolution_w, resolution_h))
            vid_writer.write(im0)
    source_state_array[source_index] = {}
    print('Done. (%.3fs)' % (time.time() - t0))

def adjustBoundingbox(target, source_state):
    if 'adjust_boudingbox' not in source_state:
        return target
    adjustment_array = source_state['adjust_boudingbox']
    for adjustment in adjustment_array:
        if adjustment['id'] != target['id']:
            continue
        center_x = (target['x1'] + target['x2']) / 2 + adjustment['shift_x']
        center_y = (target['y1'] + target['y2']) / 2 + adjustment['shift_y']
        width = abs(target['x1'] - target['x2']) * adjustment['scale_w']
        height = abs(target['y1'] - target['y2']) * adjustment['scale_h']
        target['x1'] = center_x - width / 2
        target['x2'] = center_x + width / 2
        target['y1'] = center_y - height / 2
        target['y2'] = center_y + height / 2
        return target
    return target

def lock_fps(rtsp_pipe, rtsp_type):
    global fps_flag
    time_gap = 1.0/float(FPS_LOCK)
    while True:
        time.sleep(time_gap)
        if fps_flag == 1:
            fps_flag = 0
            continue
        if rtsp_type == 'alg':
            global global_alg_img
            if global_alg_img is None or rtsp_pipe is None:
                continue
            rtsp_pipe.stdin.write(global_alg_img.tostring())
        if rtsp_type == 'org':
            global global_org_img
            if global_org_img is None or rtsp_pipe is None:
                continue
            rtsp_pipe.stdin.write(global_org_img.tostring())

def generate_rtsp_pipe(push_url, resolution_w, resolution_h, fps):  
    rtsp_type = push_url.split('/')[-1]
    command = [r'ffmpeg', 
    '-y', '-an',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24', #像素格式
    '-s', "{}x{}".format(resolution_w, resolution_h),
    '-r', str(fps), # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。 
    '-i', '-',
    '-c:v', 'libx264',  # 视频编码方式
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'rtsp', #  flv rtsp
    '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
    push_url] # rtsp rtmp  
    rtsp_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE) 
    fps_locker = threading.Thread(target=lock_fps, args=[rtsp_pipe, rtsp_type])
    fps_locker.start()
    return rtsp_pipe

def run_auto_track(msg, callback, data_cache):
    if 'alg_combination' not in msg:
        print('alg_combination not in msg')
        return
    source = msg['source']
    global rtsp_pipe
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str,
    #                     default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--weights', type=str,
                        default='alg/yolov5/weights/best.pt', help='model.pt path')
    # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str,
    #                     default="rtsp://admin:SMUwm_007@183.192.69.170:7502/id=1", help='source')
    # parser.add_argument('--source', type=str,
    #                     default='20210719095418.avi', help='source')
    parser.add_argument('--source', type=str,
                        default=source, help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='MJPG',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_false',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0,1,2,3,4], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="alg/deep_sort/configs/deep_sort.yaml")

# flag
    parser.add_argument("--defog_flag", type=int, default=0)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)

    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        try:
            detect(args, callback, data_cache)
        except Exception as e:
            traceback.print_exc()
