import numpy as np
import cv2
import os
import time

def get_geo_LS(X_, y_):
    num = len(X_)
    x2 = X_**2
    xy = X_*y_
    a = (np.sum(x2)*np.sum(y_) - np.sum(X_)*np.sum(xy))/(num*np.sum(x2) - np.sum(X_)**2)
    b = (num*np.sum(xy) - np.sum(X_)*np.sum(y_))/(num*np.sum(x2) - np.sum(X_)**2)
    return b, a

def get_leftline_deoutlier(x, y, out, out1,lr=0):
    y_max = np.max(y)
    pix = set(y[y<y_max/3])
    
    x_left = []
    y_left = []
    for i in pix:
        if lr=='r':
            x_i = sorted(x[y==i])[::-1]
        else:
            x_i = sorted(x[y==i])
        up_n = 1
        if len(x_i)>up_n:
            if (x_i[up_n]>out) |  (x_i[0]<out1):continue
            x_left += x_i[0:up_n]
            y_left += [i]*up_n
        else:
            if (x_i[-1]>out) |  (x_i[0]<out1):continue
            x_left += x_i
            y_left += [i]*len(x_i)
    return np.asarray(x_left), np.asarray(y_left)

def least_square2_leftline(bandx,bandy,left, up=0):
    X = bandx.flatten().astype('float32')
    y = bandy.flatten().astype('float32')
    up = np.max(X)/2
    X_, y_ = get_leftline_deoutlier(X, y, up, 0.0,left)
    b, a = get_geo_LS(X_, y_)
    return b, a

def coefficient_rankone_haze(data,b):
    c,d = least_square2_leftline(data[b[0]]-np.min(data[b[0]]), data[b[1]]-np.min(data[b[1]]),str(b[0])+str(b[1]), 'r')
    return c

def norm_(data):
    data_ = np.asarray(data)
    if data_.shape[0]>3:
        de=np.tile(np.linalg.norm(data_,axis=1),(3,1)).T
    else:
        de=np.linalg.norm(data_)
    data_ = data_/de
    return data_

def get_project_modif(clean_vector, cirrus_vector, point):
    point = np.asarray(point).T
    c, d = cirrus_vector[0], cirrus_vector[1]
    a, b = clean_vector[0], clean_vector[1]
    '''another clean_vector'''
    alp = np.arctan(clean_vector[1]/clean_vector[0])
    the = np.arccos(np.dot(norm_(clean_vector),norm_(cirrus_vector)))
    tan = np.tan(alp+the*2)
    cl_ = [1/tan,1]

    cl_v = np.dot(np.ones(point.shape[0]).reshape(-1,1),np.asarray(clean_vector).reshape(1,-1))
    th = point[:,0]*d-point[:,1]*c
    cl_v[np.where(th<0)[0]] = cl_
    th = a*point[:,1]-b*point[:,0]
    cl_v[np.where(th<0)[0]] = point[np.where(th<0)]

    a,b = cl_v[:,0],cl_v[:,1]
    ci = np.abs((point[:,0]*b-point[:,1]*a)/(c*b-a*d+10**(-5)))*c
    return ci

def gredient_x_r(band):
    band_l = np.zeros(band.shape)
    band_l[:-1,:] = band[1:,:]
    gre = band_l - band
    return np.abs(gre)

def gredient_y_r(band):
    band_l = np.zeros(band.shape)
    band_l[:,:-1] = band[:,1:]
    gre = band_l - band
    return np.abs(gre)


def do_DROP(frame, w, h, size, area, bright, grad, cloud_r):
    haze_part = []
    im_res_ = []
    im = frame.astype('float32')/255
    im_res = cv2.resize(im,(w,h))
    for i in range(3):           
        haze_part.append(im[:,:,i].flatten())
        im_res_.append(im_res[:,:,i])
    b = [1,0] # [1,0] [2,0] can change

    st_time = time.time()
    a = coefficient_rankone_haze(im_res_,b) 
    clean_vector =[1, np.clip(a, 0.5, 0.7)]  # 0.5 0.7 can change
    cirrus_vector = [1, 1]
    hm = [np.min(haze_part[b_]) for b_ in b]
    ci_d = get_project_modif(clean_vector, cirrus_vector, [haze_part[b[0]]-hm[0],haze_part[b[1]]-hm[1]])
    ci_d = ci_d.reshape((size[1],size[0])) if len(size)==2 else ci_d.reshape((size[0],size[1])) # video:1,0 image:0,1
    ci_d += hm[0]
    uset = time.time()-st_time
    # cv2.imwrite(os.path.join(savp,str(b[0])+str(b[1])+'cloud_modif.png'),ci_d*255)
    
    cli = np.clip(ci_d*255,0,255).astype('uint8')
    ret,bin = cv2.threshold(cli,bright,255, cv2.THRESH_BINARY)
    #cv2.imwrite(os.path.join(savp,str(b[0])+str(b[1])+'bin.png'),bin)
    kernel = np.ones((5,5), np.uint8)
    bin = cv2.erode(bin, kernel)
    kernel = np.ones((15,15), np.uint8)
    bin = cv2.dilate(bin, kernel)
    #cv2.imwrite(os.path.join(savp,str(b[0])+str(b[1])+'bin_eros.png'),bin)
    contours, hierarchy = cv2.findContours(bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    grax = gredient_x_r(im[:,:,0])
    gray = gredient_y_r(im[:,:,0])
    cloud_area = 0
    for c in range(len(contours)):
        c_area = cv2.contourArea(contours[c])
        if c_area >area:
            mask = np.zeros((size[1],size[0],3)) if len(size)==2 else np.zeros(size)
            cv2.drawContours(mask, contours, c,(255, 0, 0), thickness=-1)
            # cv2.imwrite(os.path.join(savp,str(b[0])+str(b[1])+'cont.png'),mask)
            inx = mask[:,:,0]==255
            c_grad = np.mean(grax[inx]+gray[inx])
            print(c_grad)
            if c_grad < grad:
                frame = cv2.drawContours(frame, contours, c, (0, 0, 255), 2) #frame
                cloud_area += c_area
                # cv2.imwrite(os.path.join(savp,str(b[0])+str(b[1])+'cloud_cont.png'),frame)
    cloud_rat = cloud_area/(size[0]*size[1])
    print(f'=============================cloud_area {cloud_area}')
    if cloud_rat > cloud_r:
        print('cloud ratio is: %.3f, there is cloud layer.' % cloud_rat)
    else:
        print('cloud ratio is: %.3f, there is not cloud layer.' % cloud_rat)
    return cloud_area, cloud_rat > cloud_r

def DROP_image(bright, grad, cloud_r, input, savp):
    image = cv2.imread(input)
    size = image.shape # 
    scale = np.ceil(np.sqrt(size[0]*size[1]/18000)) # time sort: change 18000 lower
    w, h = int(size[0]/scale),int(size[1]/scale)
    area = size[0] # when cloud area is higher than 'are', it maybe cloud;
    do_DROP(image, w, h, size, area, bright, grad, cloud_r)
    return

def DROP_video(bright, grad, cloud_r, vedio_input, vedio_out, savp):
    cap = cv2.VideoCapture(vedio_input)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # video use: size[0] 
    suc = cap.isOpened()
    frame_count = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(vedio_out, fourcc, frame_count, size)

    scale = np.ceil(np.sqrt(size[0]*size[1]/18000)) # time sort: change 18000 lower
    w, h = int(size[0]/scale),int(size[1]/scale)
    area = size[0] # when cloud area is higher than 'are', it maybe cloud;
    while suc:
        suc, frame = cap.read()
        if suc :
            do_DROP(frame, w, h, size,area, bright, grad, cloud_r)
            videoWriter.write((frame).astype('uint8'))
            #break
    videoWriter.release()
    cap.release()
    return

def cloud_det(image, size):
    scale = np.ceil(np.sqrt(size[0]*size[1]/18000)) # time sort: change 18000 lower
    w, h = int(size[0]/scale),int(size[1]/scale)
    area = size[0] # when cloud area is higher than 'are', it maybe cloud;
    bright_th = 100 # cloud-map(ci_d)'s pixel value higher than 'bright_th' maybe cloud
    grad_th = 0.05 # when the gradient of cloud area is lower than 'grad_th', it maybe cloud
    cloud_ratio = 0.1 # if cloud's ratio is higher than 'cloud_ratio', there is cloud layer
    area, res = do_DROP(image, w, h, size,area, bright_th, grad_th, cloud_ratio)
    return area, res

if __name__ == '__main__':
    size = (906, 604)
    image = cv2.imread('defog.jpg')
    scale = np.ceil(np.sqrt(size[0]*size[1]/18000)) # time sort: change 18000 lower
    w, h = int(size[0]/scale),int(size[1]/scale)
    area = size[0] # when cloud area is higher than 'are', it maybe cloud;
    bright_th = 100 # cloud-map(ci_d)'s pixel value higher than 'bright_th' maybe cloud
    grad_th = 0.05 # when the gradient of cloud area is lower than 'grad_th', it maybe cloud
    cloud_ratio = 0.1 # if cloud's ratio is higher than 'cloud_ratio', there is cloud layer
    res = do_DROP(image, w, h, size,area, bright_th, grad_th, cloud_ratio)
    print(res)