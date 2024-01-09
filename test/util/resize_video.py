import os
import time
import cv2 

def resize_video(input_path, output_path, target_reslution):
    input_video = cv2.VideoCapture(input_path)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    # width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, target_reslution)
    while input_video.isOpened():
        ret, frame = input_video.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, target_reslution)
        output_video.write(resized_frame)
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    root_directory = 'C:/Users/hasee/Desktop/rtsp'
    for root, dir, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):
                input_path = root + '/' + file
                output_path = root + '/o_' + file.split('.')[0] + '.mp4'
                resize_video(input_path, output_path, (1280, 720))

