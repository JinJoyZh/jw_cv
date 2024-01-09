import time
import cv2 

target_reslution = (1920, 1080)

a = [1,2,3]

def change(a):
    b = a
    b[0] = 111

if __name__ == "__main__":
    a = int(time.time() * 1000)
    a = [1,2,3,4]
    print(id(a))