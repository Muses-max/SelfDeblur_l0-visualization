import torch
from utils.common_utils import *
import numpy as np
import os
from networks.skip import skip
from networks.fcn import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
import cv2
from copy import deepcopy
from math import sqrt

INPUT = 'noise' #input type
pad = 'reflection'
impath = 'Blurry1_8.png'
grad = None
n_k = 200
factor = 1.4
initial_image_size = 256
input_depth = 8
channels = 1
initial_kernel_size = 39
max_iter = 5000
save = 200
scales = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = %s"%device)
reg_noise_std = 0.001
LR = 0.02
l = True if channels == 1 else False
patchsize = 35

kernel_size = [initial_kernel_size, initial_kernel_size]
img_size = [initial_image_size, initial_image_size]
input_size = (img_size[0] + kernel_size[0] - 1, img_size[1] + kernel_size[1] - 1)
scale = 1
pixelloss = torch.nn.MSELoss()
ssim = SSIM()
gradop = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
lap = torch.from_numpy(np.array(gradop).astype(np.float32)).to(device)
lap = lap.unsqueeze(0).unsqueeze(0)
if channels > 1:
    lap = torch.cat([lap]*channels, 0)
lap.requires_grad = False
lap.trainable = False
ratio = 1.1
max_g_weight = 16
latest = 3000
dc_weight = 0.1
while scale <= scales:
    l0_weight = 1
    g_weight = ratio*l0_weight
    print("current_scale = %s, ratio = %s"%(scale, ratio))
    net = skip(input_depth, channels,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').to(device)
    im, im_np, im_torch = im_to_torch(impath, imsize=img_size[0], L=l, grad=grad)
    net_input_kernel = get_noise(n_k, INPUT, (1, 1))
    net_input_kernel.squeeze_()
    net_input_kernel = net_input_kernel.to(device)
    net_kernel = fcn(n_k, kernel_size[0] * kernel_size[1]).to(device)
    net_input = get_noise(input_depth, INPUT, input_size).to(device)
    if scale == 1:
        kernel_pruned = torch.zeros((kernel_size[0], kernel_size[1])).to(device)
    else:
        kernel_pruned = torch.from_numpy(kernel_refined).to(device)
    kernel_pruned.requires_grad = False
    kernel_pruned.trainable = False
        
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)        
    net_input_saved = net_input
    pad0, pad1 = (kernel_size[0] - 1)//2, (kernel_size[1] - 1)//2
    crop = [
        [pad0, pad0+img_size[0]],
        [pad1, pad1+img_size[1]],
     ]
    for cnt in range(1, max_iter+1):
        net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).normal_().to(device)
        
        # get the network output
        out_x = net(net_input)
        if cnt < latest:
            gradient = nn.functional.conv2d(out_x, lap, padding=1, bias=None, groups=channels)
            g = gradient.detach()
            g[gradient < sqrt(l0_weight/g_weight)] = 0.0
        absgrad = torch.abs(gradient)
        l0 = torch.sum(absgrad > 0.0)
        out_k = net_kernel(net_input_kernel).view(-1,1,kernel_size[0],kernel_size[1])
        if scale == 1:
            out_k_m = out_k
        else:
            out_k_m = 0.5*(out_k + kernel_pruned)
        if channels > 1:
            out_k_m = torch.cat([out_k_m]*channels, 0)
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None, groups=channels)
        #print(out_y.shape, im_torch.shape)
        if cnt < latest:
            loss = pixelloss(out_y, im_torch) + g_weight * pixelloss(gradient, g)
        else:
            loss = pixelloss(out_y, im_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(cnt)
        
        if cnt % save == 0:
            out_vis = torch_to_im(out_y, '%s_conv.png'%scale, grad=grad)
            im_vis = torch_to_im(im_torch, '%s_gt.png'%scale,grad=grad)
            torch_to_im(out_x, '%s_result.png'%(scale), crop=crop,grad=grad)
            visualize_kernel(out_k_m, '%s_kernel.png'%(scale))
            print("scale = %s, cnt = %s, mse = %s"%(scale, cnt, loss))
            print("l0 = %s"%l0)
        if g_weight < max_g_weight: 
            g_weight *= ratio
        if cnt >= latest:
            g_weight = 0.0
        
            
    out_x = out_x[:,:, pad0:pad0+img_size[0], pad1:pad1+img_size[1]]
    kernel_tmp = out_k_m[0][0].detach().cpu().numpy()
    kernel = deepcopy(kernel_tmp)
    kernel_refined = refine(kernel_tmp)
    net = skip(input_depth, channels,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').to(device)
    kernel_refined_torch = torch.from_numpy(kernel_refined).to(device).view(-1, 1, kernel_size[0], kernel_size[1])
    net_input = get_noise(input_depth, INPUT, input_size).to(device)
    optimizer = torch.optim.Adam([{'params':net.parameters()}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.4)   
    for cnt in range(1, (max_iter+1)//5):
        out_x = net(net_input)
        out_y = nn.functional.conv2d(out_x, kernel_refined_torch, padding=0, bias=None, groups=channels)
        loss = pixelloss(out_y, im_torch) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(cnt)
        if cnt % save == 0:
            out_vis = torch_to_im(out_y, '%s_conv.png'%scale, grad=grad)
            torch_to_im(out_x, '%s_result.png'%(scale), crop=crop,grad=grad)
            print(cnt, "nonblind deconvolution")
            
    write2Darray(kernel, path='original.txt')
    kernel_refined = cv2.resize(kernel_refined, tuple(kernel_size), interpolation=cv2.INTER_CUBIC)
    kernel_refined[kernel_refined < 0.0] = 0.0
    kernel_refined /= np.sum(kernel_refined)
    kernel_size = list(map(lambda x: 2*x+1, kernel_size))
    img_size = list(map(lambda x:2*x, img_size))
    input_size = (img_size[0] + kernel_size[0] - 1, img_size[1] + kernel_size[1] - 1)
    scale += 1
    n_k = int(n_k*factor)
print("Done.")