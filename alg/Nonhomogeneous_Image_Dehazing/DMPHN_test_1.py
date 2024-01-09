import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import io
import os
import argparse
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time
from PIL import Image
import cv2

sys.path.append('..')
from Nonhomogeneous_Image_Dehazing import models

#Hyper Parameters
METHOD = "DMPHN_1_2_4"

# SAMPLE_DIR = "Nonhomogeneous_Image_Dehazing/image"       # store hazy images here

GPU = 0

def load_pkl():
    print("init data folders")
    # print(image_name)

    encoder_lv1 = models.Encoder().cuda(GPU)
    encoder_lv2 = models.Encoder().cuda(GPU)
    encoder_lv3 = models.Encoder().cuda(GPU)

    decoder_lv1 = models.Decoder().cuda(GPU)
    decoder_lv2 = models.Decoder().cuda(GPU)
    decoder_lv3 = models.Decoder().cuda(GPU)

    encoder_lv1.load_state_dict(torch.load(str('alg/Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/encoder_lv1.pkl")))
    encoder_lv2.load_state_dict(torch.load(str('alg/Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/encoder_lv2.pkl")))
    encoder_lv3.load_state_dict(torch.load(str('alg/Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/encoder_lv3.pkl")))

    decoder_lv1.load_state_dict(torch.load(str('alg/Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/decoder_lv1.pkl")))
    decoder_lv2.load_state_dict(torch.load(str('alg/Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/decoder_lv2.pkl")))
    decoder_lv3.load_state_dict(torch.load(str('alg/Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/decoder_lv3.pkl")))

    return encoder_lv1, encoder_lv2, encoder_lv3, decoder_lv1, decoder_lv2, decoder_lv3


def NID_main(image, encoder_lv1, encoder_lv2, encoder_lv3, decoder_lv1, decoder_lv2, decoder_lv3):

    # print("init data folders")
    # # print(image_name)
    #
    # encoder_lv1 = models.Encoder().cuda(GPU)
    # encoder_lv2 = models.Encoder().cuda(GPU)
    # encoder_lv3 = models.Encoder().cuda(GPU)
    #
    # decoder_lv1 = models.Decoder().cuda(GPU)
    # decoder_lv2 = models.Decoder().cuda(GPU)
    # decoder_lv3 = models.Decoder().cuda(GPU)
    #
    # encoder_lv1.load_state_dict(torch.load(str('Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/encoder_lv1.pkl")))
    # encoder_lv2.load_state_dict(torch.load(str('Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/encoder_lv2.pkl")))
    # encoder_lv3.load_state_dict(torch.load(str('Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/encoder_lv3.pkl")))
    #
    # decoder_lv1.load_state_dict(torch.load(str('Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/decoder_lv1.pkl")))
    # decoder_lv2.load_state_dict(torch.load(str('Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/decoder_lv2.pkl")))
    # decoder_lv3.load_state_dict(torch.load(str('Nonhomogeneous_Image_Dehazing/checkpoints/' + METHOD + "/decoder_lv3.pkl")))

    # os.makedirs('Nonhomogeneous_Image_Dehazing/image_out/', exist_ok = True)

    iteration = 0.0
    test_time = 0.0
    

    with torch.no_grad():
        images_lv1 = transforms.ToTensor()(image)    # Image.open(image).convert('RGB')
        images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).cuda(GPU)
        # images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0)

        start = time.time()
        H = images_lv1.size(2)
        W = images_lv1.size(3)

        images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
        images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        feature_lv3_1 = encoder_lv3(images_lv3_1)
        feature_lv3_2 = encoder_lv3(images_lv3_2)
        feature_lv3_3 = encoder_lv3(images_lv3_3)
        feature_lv3_4 = encoder_lv3(images_lv3_4)
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        residual_lv3_top = decoder_lv3(feature_lv3_top)
        residual_lv3_bot = decoder_lv3(feature_lv3_bot)

        feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
        residual_lv2 = decoder_lv2(feature_lv2)

        feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
        dehazed_image = decoder_lv1(feature_lv1)

        stop = time.time()
        test_time += stop-start
        print('RunTime:%.4f'%(stop-start), '  Average Runtime:%.4f'%(test_time/(iteration+1)))

        # kll = dehazed_image.data + 0.5
        # kll = kll.squeeze().cpu().numpy()
        # kll = (kll * 255).astype(np.uint8)
        # kll = kll.transpose((1, 2, 0))
        # kll = cv2.cvtColor(kll, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('kll.jpg', kll)

        # kll = torchvision.utils.save_image(dehazed_image.data + 0.5, image)

        grid = torchvision.utils.make_grid(dehazed_image.data + 0.5)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        # im = Image.fromarray(ndarr)

    return ndarr

# if __name__ == '__main__':
#     image = 'input/47.png'
#     NID_main(image)