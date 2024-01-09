import cv2

start_time = 80
end_time = 106
input_path = r'C:/Users/hasee/Desktop/rtsp/1.mp4'
output_path = r'C:/Users/hasee/Desktop/rtsp/output.mp4'
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame_index = int (start_time * fps)
end_frame_index = int(end_time * fps)
if start_frame_index < 0:
    start_frame_index = 0
if end_frame_index > frame_count:
    end_frame_index = frame_count
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
for frame in range(start_frame_index, end_frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame = cap.read()
    if ret:
        out.write(frame)
    else:
        print("unable to read", frame)
cap.release()
out.release()
cv2.destroyAllWindows()