import cv2
import math
import numpy as np
import time
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel) # find min value in kernel range
    return dark

def DarkChannel_sphere(im,sz):
    im2 = np.power(im,2)
    im2_mean = cv2.boxFilter(im2,ddepth=-1,ksize=(sz,sz),normalize=True)
    im_mean = cv2.boxFilter(im,ddepth=-1,ksize=(sz,sz),normalize=True)
    im_mean2 = np.power(im_mean,2)
    sigm = np.tile(np.mean(np.sqrt(im2_mean-im_mean2),axis=2)[:,:,np.newaxis],(1,1,3))
    dark = np.min(im_mean - sigm,axis=2)
    return dark

def DarkChannel_c(im,sz):
    b,g,r = cv2.split(im)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    darkb = cv2.erode(b,kernel) # find min value in kernel range
    darkg = cv2.erode(g,kernel)
    darkr = cv2.erode(r,kernel)
    return darkb, darkg, darkr

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1)) #0.1%
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::] # value after index 40000-40: 

    atmsum = np.zeros([1,3])
    for ind in range(0,numpx): # 1
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(im,A,sz,c=0):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    if not c:
        transmission = 1 - omega*DarkChannel(im3,sz)
        return transmission
    else:
        darkb, darkg, darkr = DarkChannel_c(im3,sz)
        tb = 1 - omega*darkb
        tg = 1 - omega*darkg
        tr = 1 - omega*darkr
        return tb, tg, tr

def TransmissionEstimate_Sphere(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel_sphere(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res/np.amax(res)

def Recover_c(im,t,A):
    res = np.empty(im.shape,im.dtype)
    #t = [cv2.max(tc,tx) for tc in t] #

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t[ind] + A[0,ind]
    return res/np.amax(res)

def stretch(band,p):
    lowband_nozero=np.percentile(band,p)#0.001   p
    upband_nozero=np.percentile(band,100-p)
    wu = np.where(band>=upband_nozero)
    wl = np.where(band<=lowband_nozero)
    bandgray=np.clip(band,a_min=lowband_nozero,a_max=upband_nozero)
    if (upband_nozero-lowband_nozero)!=0:
        band_clip=(bandgray-lowband_nozero)/(upband_nozero-lowband_nozero)
    else:
        band_clip=bandgray
    return band_clip

def stretch_J(J):
    sj = np.empty(J.shape, J.dtype)
    for i in range(3):
        sj[:,:,i] = stretch(J[:,:,i], 0.1)
    return sj

def do_DCP(data,sdcp):
    ker = 15
    dark = DarkChannel(data,ker)#15
    A = AtmLight(data,dark)
    if sdcp:
        te = TransmissionEstimate_Sphere(data,A,ker)
    else:
        te = TransmissionEstimate(data,A,ker)
    t = TransmissionRefine(data,te)
    J = Recover(data,t,A,0.1).transpose(1,0,2)
    sj = stretch_J(J)
    return sj,J

def homomorphic_filter(src, d0=5, r1=0.25, rh=2, c=8, h=1.0, l=0.):
    gray = src.copy()
    if len(src.shape) > 2:#维度>2
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    gray = np.float64(gray) 
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray) 
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)

    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows//2, rows//2))#

    D = np.sqrt(M ** 2 + N ** 2)#
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l

    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    return dst

def SDCP(image):
    I = image.astype('float32')/255
    homo_I = np.zeros(I.shape).astype('float32')
    for i in range(3):
        homo_I[:,:,i] = homomorphic_filter(I[:,:,i], d0=4, r1=0.5, rh=2, c=4, h=2.0, l=0.5)
    #sj,J = do_DCP(I,sdcp=0)
    shj,HJ = do_DCP(homo_I,sdcp=1)
    sav_im = np.clip(cv2.flip(cv2.rotate(shj,cv2.ROTATE_90_CLOCKWISE),1)*255,0,255)#*2.5
    return sav_im.astype('uint8')


    