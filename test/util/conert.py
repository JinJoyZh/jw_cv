import cv2

input_path = r'C:/Users/hasee/Desktop/tmp/1.avi'
output_path = r'C:/Users/hasee/Desktop/tmp/out.mp4'
def convert4mp4(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
