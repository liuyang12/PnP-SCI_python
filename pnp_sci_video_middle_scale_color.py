# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 08:46:46 2020

@author: Xin
"""


#!/usr/bin/env python
# coding: utf-8

# ## Deep Video Priors for Snapshot Compressive Imaging
#  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, MIT CSAIL"), [MIT CSAIL](https://www.csail.mit.edu/), yliu@csail.mit.edu, updated Dec 9, 2019.

# In[1]:


import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean

from pnp_sci import gap_denoise_bayer

# In[2]:


# [0] environment configuration
# datasetdir = './dataset/cacti' # dataset
datasetdir = './dataset/cacti/middle_scale' # dataset
resultsdir = './results' # results

# datname = 'kobe' # name of the dataset
# datname = 'traffic' # name of the dataset
# datname = 'bus_bayer' # name of the dataset
# datname = 'bus_256_bayer' # name of the dataset
# datname = 'traffic_bayer' # name of the dataset#
for ncount in range(1):
#datname = 'Traffic_bayer'
# datname = 'Runner_bayer'
# datname = 'Bosphorus_bayer'
    if ncount == 0:
        datname = 'Bosphorus_bayer'
    elif ncount == 1:
        datname = 'Traffic_bayer'
    elif ncount == 2:
        datname = 'Beauty_bayer'
    elif ncount == 3:
        datname = 'Runner_bayer'
    elif ncount == 4:
        datname = 'ShakeNDry_bayer'
    elif ncount == 5:
        datname = 'Jockey_bayer'
    else:
        datname = 'Jockey_bayer'    
    #datname = 'ShakeNDry_bayer'
    # datname = 'messi_bayer' # name of the dataset
    # datname = 'messi_c24_bayer' # name of the dataset
    # datname = 'hummingbird_c40_bayer' # name of the dataset
    # datname = 'football_cif_bayer' # name of the dataset
    # datname = 'test' # name of the dataset
    # varname = 'X' # name of the variable in the .mat data file
    
    matfile = datasetdir + '/' + datname + '.mat' # path of the .mat data file
    
    
    # In[3]:
    
    
    # [1] load data
    with h5py.File(matfile, 'r') as file: # for '-v7.3' .mat file (MATLAB)
        # print(list(file.keys()))
        meas_bayer = np.array(file['meas_bayer'])
        mask_bayer = np.array(file['mask_bayer'])
        orig_bayer = np.array(file['orig_bayer'])
        orig_real = np.array(file['orig'])
    #==============================================================================
    # file = scipy.io.loadmat(matfile) # for '-v7.2' and below .mat file (MATLAB)
    # X = list(file[varname])
    #file = sio.loadmat(matfile)
    #meas_bayer = np.array(file['meas'])
    #mask_bayer = np.array(file['mask'])
    #orig_bayer = np.array(file['orig_bayer'])
    
    #==============================================================================
    
    mask_bayer = np.float32(mask_bayer).transpose((2,1,0))
    if len(meas_bayer.shape) < 3:
        meas_bayer = np.float32(meas_bayer).transpose((1,0))
    else:
        meas_bayer = np.float32(meas_bayer).transpose((2,1,0))
    orig_bayer = np.float32(orig_bayer).transpose((2,1,0))
    # print(mask_bayer.shape, meas_bayer.shape, orig_bayer.shape)
    (nrows, ncols,nmea) = meas_bayer.shape
    (nrows, ncols,nmask) = mask_bayer.shape
    
    vgaptv_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_gaptv = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_gaptv = np.zeros([nmask*nmea,1], dtype=np.float32)
    
    vgapffdgray_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_gapffdgray = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_gapffdgray = np.zeros([nmask*nmea,1], dtype=np.float32)
    
    vgap_ffd_color_down_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_gapffd_color_down = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_gapffd_color_down = np.zeros([nmask*nmea,1], dtype=np.float32)
    
    vgap_ffd_color_demosaic_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_gapffd_color_demosaic = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_gapffd_color_demosaic = np.zeros([nmask*nmea,1], dtype=np.float32)
    
    vgap_fastdvd_gray_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_fastdvd_gray = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_fastdvd_gray = np.zeros([nmask*nmea,1], dtype=np.float32)
    
    vgap_fastdvd_color_down_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_fastdvd_color_down = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_fastdvd_color_down = np.zeros([nmask*nmea,1], dtype=np.float32)
    
    vgap_fastdvd_color_demosaic_bayer = np.zeros([nrows, ncols, nmask*nmea], dtype=np.float32)
    psnr_fastdvd_color_demosaic = np.zeros([nmask*nmea,1], dtype=np.float32)
    ssim_fastdvd_color_demosaic = np.zeros([nmask*nmea,1], dtype=np.float32)
    
    for iframe in range(nmea):
        MAXB = 255.
        
        if len(meas_bayer.shape) >= 3:
            meas_bayer_t = np.squeeze(meas_bayer[:,:,iframe])/MAXB
        else:
            meas_bayer_t = meas_bayer/MAXB
        orig_bayer_t = orig_bayer[:,:,iframe*nmask:(iframe+1)*nmask]/MAXB
        
        
        # In[4]:
        
        
        ## [2.1] GAP-TV [for baseline reference]
        _lambda = 1 # regularization factor
        accelerate = True # enable accelerated version of GAP
        denoiser = 'tv' # total variation (TV)
        iter_max = 40 # maximum number of iterations
        tv_weight = 0.1 # TV denoising weight (larger for smoother but slower)
        tv_iter_max = 5 # TV denoising maximum number of iterations each
        begin_time = time.time()
        vgaptv_bayer_t,psnr_gaptv_t,ssim_gaptv_t,psnrall_tv =             gap_denoise_bayer(meas_bayer_t, mask_bayer, _lambda, 
                                      accelerate, denoiser, iter_max, 
                                      tv_weight=tv_weight, 
                                      tv_iter_max=tv_iter_max,
                                      X_orig=orig_bayer_t)
        end_time = time.time()
        tgaptv = end_time - begin_time
        print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            denoiser.upper(), mean(psnr_gaptv_t), mean(ssim_gaptv_t), tgaptv))
        vgaptv_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  vgaptv_bayer_t
        psnr_gaptv[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_gaptv_t,(nmask,1))
        ssim_gaptv[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_gaptv_t,(nmask,1))
        
        
        # In[9]:    
        import torch
        from packages.ffdnet.models import FFDNet
        
        ## [2.5] GAP/ADMM-FFDNet
        ### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
        _lambda = 1 # regularization factor
        accelerate = True # enable accelerated version of GAP
        denoiser = 'ffdnet' # video non-local network 
        noise_estimate = False # disable noise estimation for GAP
        # sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
        # iter_max = [20,20,20,10] # maximum number of iterations
        # sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
        # iter_max = [10,10,10,10] # maximum number of iterations
        sigma    = [50/255,25/255, 12/255, 6/255] # pre-set noise standard deviation
        iter_max = [20,20,20,20] # maximum number of iterations
        useGPU = True # use GPU
        
        # pre-load the model for FFDNet image denoising
        in_ch = 1
        model_fn = 'packages/ffdnet/models/net_gray.pth'
        # Absolute path to model file
        # model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)
        
        # Create model
        net = FFDNet(num_input_channels=in_ch)
        # Load saved weights
        if useGPU:
            state_dict = torch.load(model_fn)
            device_ids = [0]
            model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
        else:
            state_dict = torch.load(model_fn, map_location='cpu')
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_dataparallel_wrapper(state_dict)
            model = net
        model.load_state_dict(state_dict)
        model.eval() # evaluation mode
        
        begin_time = time.time()
        vgapffdnet_bayer_t,psnr_gapffdnet_t,ssim_gapffdnet_t,psnrall_ffdnet =                 gap_denoise_bayer(meas_bayer_t, mask_bayer, _lambda, 
                                          accelerate, denoiser, iter_max, 
                                          noise_estimate, sigma,
                                          x0_bayer=None,
                                          X_orig=orig_bayer_t,
                                          model=model)
        end_time = time.time()
        tgapffdnet = end_time - begin_time
        print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f} running time {:.1f} seconds.'.format(
            denoiser.upper(), mean(psnr_gapffdnet_t), mean(ssim_gapffdnet_t), tgapffdnet))
        vgapffdgray_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  vgapffdnet_bayer_t
        psnr_gapffdgray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_gapffdnet_t,(nmask,1))
        ssim_gapffdgray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_gapffdnet_t,(nmask,1))
        
        # In[10]:
        
        
        import torch
        from packages.ffdnet.models import FFDNet
        
        ## [2.5] GAP/ADMM-FFDNet
        ### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
        _lambda = 1 # regularization factor
        accelerate = True # enable accelerated version of GAP
        denoiser = 'ffdnet_color_down' # video non-local network 
        noise_estimate = False # disable noise estimation for GAP
        # sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
        # iter_max = [20,20,20,10] # maximum number of iterations
        # sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
        # iter_max = [10,10,10,10] # maximum number of iterations
        sigma    = [50/255,25/255,12/255,6/255] # pre-set noise standard deviation
        iter_max = [20,20,20,10] # maximum number of iterations
        useGPU = True # use GPU
        
        # pre-load the model for FFDNet image denoising
        model_name = 'ffdnet_color'           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
          
            #sf = 1                    # unused for denoising
        if 'color' in model_name:
                n_channels = 3        # setting for color image
                nc = 96               # setting for color image
                nb = 12               # setting for color image
        else:
                n_channels = 1        # setting for grayscale image
                nc = 64               # setting for grayscale image
                nb = 15               # setting for grayscale image
            
        model_pool = 'packages/ffdnet/models'  # fixed
        model_path = os.path.join(model_pool, model_name+'.pth')
            #model_path = os.path.join(model_pool, model_name+'.pth')
        
            # ----------------------------------------
          
            #need_H = True if H_path is not None else False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        
            # ----------------------------------------
            # load model
            # ----------------------------------------
        
            #from packages.ffdnet.models.network_ffdnet import FFDNet as net
        from packages.ffdnet.network_ffdnet import FFDNet as net
        model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
                v.requires_grad = False
        model = model.to(device)
            
        
        begin_time = time.time()
        vgapffdnet_color_down_bayer_t,psnr_gapffdnet_color_down_t,ssim_gapffdnet_color_down_t,psnrall_ffdnet_color_down =                 gap_denoise_bayer(meas_bayer_t, mask_bayer, _lambda, 
                                          accelerate, denoiser, iter_max, 
                                          noise_estimate, sigma,
                                          x0_bayer=vgaptv_bayer_t,
                                          X_orig=orig_bayer_t,
                                          model=model)
        end_time = time.time()
        tgapffdnet_color_down = end_time - begin_time
        print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f} running time {:.1f} seconds.'.format(
            denoiser.upper(), mean(psnr_gapffdnet_color_down_t), mean(ssim_gapffdnet_color_down_t), tgapffdnet_color_down))
        vgap_ffd_color_down_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  vgapffdnet_color_down_bayer_t
        psnr_gapffd_color_down[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_gapffdnet_color_down_t,(nmask,1))
        ssim_gapffd_color_down[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_gapffdnet_color_down_t,(nmask,1))
        
        # In[11]:
        
        
        import torch
        from packages.ffdnet.models import FFDNet
        
        ## [2.5] GAP/ADMM-FFDNet
        ### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
        _lambda = 1 # regularization factor
        accelerate = True # enable accelerated version of GAP
        denoiser = 'ffdnet_color_demosaic' # video non-local network 
        noise_estimate = False # disable noise estimation for GAP
        # sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
        # iter_max = [20,20,20,10] # maximum number of iterations
        # sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
        # iter_max = [10,10,10,10] # maximum number of iterations
        #sigma    = [50/255,25/255,12/255,6/255] # pre-set noise standard deviation
        #iter_max = [20,20,20,10] # maximum number of iterations
        sigma    = [50/255,25/255,12/255,6/255] # pre-set noise standard deviation
        iter_max = [20,20,20,10] # maximum number of iterations
        useGPU = True # use GPU
        
        # pre-load the model for FFDNet image denoising
        model_name = 'ffdnet_color'           # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
          
            #sf = 1                    # unused for denoising
        if 'color' in model_name:
                n_channels = 3        # setting for color image
                nc = 96               # setting for color image
                nb = 12               # setting for color image
        else:
                n_channels = 1        # setting for grayscale image
                nc = 64               # setting for grayscale image
                nb = 15               # setting for grayscale image
            
        model_pool = 'packages/ffdnet/models'  # fixed
        model_path = os.path.join(model_pool, model_name+'.pth')
            #model_path = os.path.join(model_pool, model_name+'.pth')
        
            # ----------------------------------------
          
            #need_H = True if H_path is not None else False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        
            # ----------------------------------------
            # load model
            # ----------------------------------------
        
            #from packages.ffdnet.models.network_ffdnet import FFDNet as net
        from packages.ffdnet.network_ffdnet import FFDNet as net
        model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
                v.requires_grad = False
        model = model.to(device)
            
        
        begin_time = time.time()
        vgapffdnet_color_demosaic_bayer_t,psnr_gapffdnet_color_demosaic_t,ssim_gapffdnet_color_demosaic_t,psnrall_ffdnet_color =                 gap_denoise_bayer(meas_bayer_t, mask_bayer, _lambda, 
                                          accelerate, denoiser, iter_max, 
                                          noise_estimate, sigma,
                                          x0_bayer=vgaptv_bayer_t,
                                          X_orig=orig_bayer_t,
                                          model=model)
        end_time = time.time()
        tgapffdnet_color_demosaic = end_time - begin_time
        print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f} running time {:.1f} seconds.'.format(
            denoiser.upper(), mean(psnr_gapffdnet_color_demosaic_t), mean(ssim_gapffdnet_color_demosaic_t), tgapffdnet_color_demosaic))
        vgap_ffd_color_demosaic_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  vgapffdnet_color_demosaic_bayer_t
        psnr_gapffd_color_demosaic[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_gapffdnet_color_demosaic_t,(nmask,1))
        ssim_gapffd_color_demosaic[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_gapffdnet_color_demosaic_t,(nmask,1))
    
        
        # In[11]:
        
        import torch
        from packages.fastdvdnet.models import FastDVDnet
        
        ## [2.2] GAP-FastDVDnet
        _lambda = 1 # regularization factor
        accelerate = True # enable accelerated version of GAP
        denoiser = 'fastdvdnet_gray' # video non-local network 
        noise_estimate = False # disable noise estimation for GAP
        # sigma    = [50/255, 25/255, 12/255, 6/255, 3/255] # pre-set noise standard deviation
        # iter_max = [10, 10, 10, 10, 10] # maximum number of iterations
        sigma    = [50/255, 25/255, 12/255] # pre-set noise standard deviation
        iter_max = [20, 20, 20] # maximum number of iterations
        # sigma    = [50/255,25/255] # pre-set noise standard deviation
        # iter_max = [10,10] # maximum number of iterations
        useGPU = True # use GPU
        
        # pre-load the model for FFDNet image denoising
        NUM_IN_FR_EXT = 5 # temporal size of patch
        model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT,num_color_channels=1)
    
        
        # Load saved weights
        state_temp_dict = torch.load('./packages/fastdvdnet/model_gray.pth')
        if useGPU:
            device_ids = [0]
            #model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
            model = model.cuda()
        else:
            # CPU mode: remove the DataParallel wrapper
            state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
        model.load_state_dict(state_temp_dict)
        
        # Sets the model in evaluation mode (e.g. it removes BN)
        model.eval()
        
        begin_time = time.time()
        vgapfastdvdnet_gray_bayer_t,psnr_gapfastdvdnet_gray_t,ssim_gapfastdvdnet_gray_t,psnrall_fastdvdnet_gray_t =                 gap_denoise_bayer(meas_bayer_t, mask_bayer, _lambda, 
                                          accelerate, denoiser, iter_max, 
                                          noise_estimate, sigma,
                                          x0_bayer=vgaptv_bayer_t,
                                          X_orig=orig_bayer_t,
                                          model=model)
        end_time = time.time()
        tgapfastdvdnet_gray = end_time - begin_time
        print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            denoiser.upper(), mean(psnr_gapfastdvdnet_gray_t), mean(ssim_gapfastdvdnet_gray_t), tgapfastdvdnet_gray))
    
        vgap_fastdvd_gray_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  vgapfastdvdnet_gray_bayer_t
        psnr_fastdvd_gray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_gapfastdvdnet_gray_t,(nmask,1))
        ssim_fastdvd_gray[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_gapfastdvdnet_gray_t,(nmask,1))
        # In[12]:
        import torch
        from packages.fastdvdnet.models import FastDVDnet
        
        ## [2.2] GAP-FastDVDnet
        _lambda = 1 # regularization factor
        accelerate = True # enable accelerated version of GAP
        denoiser = 'fastdvdnet_down' # video non-local network 
        noise_estimate = False # disable noise estimation for GAP
        # sigma    = [50/255, 25/255, 12/255, 6/255, 3/255] # pre-set noise standard deviation
        # iter_max = [10, 10, 10, 10, 10] # maximum number of iterations
        sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
        iter_max = [20, 20, 20, 20] # maximum number of iterations
        # sigma    = [50/255,25/255] # pre-set noise standard deviation
        # iter_max = [10,10] # maximum number of iterations
        useGPU = True # use GPU
        
        # pre-load the model for FFDNet image denoising
        NUM_IN_FR_EXT = 5 # temporal size of patch
        model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
        
        # Load saved weights
        state_temp_dict = torch.load('./packages/fastdvdnet/model.pth')
        if useGPU:
            device_ids = [0]
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            # CPU mode: remove the DataParallel wrapper
            state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
        model.load_state_dict(state_temp_dict)
        
        # Sets the model in evaluation mode (e.g. it removes BN)
        model.eval()
        
        begin_time = time.time()
        vgapfastdvdnet_bayer_t,psnr_gapfastdvdnet_t,ssim_gapfastdvdnet_t,psnrall_fastdvdnet_t =                 gap_denoise_bayer(meas_bayer_t, mask_bayer, _lambda, 
                                          accelerate, denoiser, iter_max, 
                                          noise_estimate, sigma,
                                          x0_bayer=vgaptv_bayer_t,
                                          X_orig=orig_bayer_t,
                                          model=model)
        end_time = time.time()
        tgapfastdvdnet_down = end_time - begin_time
        print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            denoiser.upper(), mean(psnr_gapfastdvdnet_t), mean(ssim_gapfastdvdnet_t), tgapfastdvdnet_down))
    
        vgap_fastdvd_color_down_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  vgapfastdvdnet_bayer_t
        psnr_fastdvd_color_down[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_gapfastdvdnet_t,(nmask,1))
        ssim_fastdvd_color_down[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_gapfastdvdnet_t,(nmask,1))
    
        # In[13]:
        import torch
        from packages.fastdvdnet.models import FastDVDnet
        
        ## [2.2] GAP-FastDVDnet
        _lambda = 1 # regularization factor
        accelerate = True # enable accelerated version of GAP
        denoiser = 'fastdvdnet_demosaic' # video non-local network 
        noise_estimate = False # disable noise estimation for GAP
        # sigma    = [50/255, 25/255, 12/255, 6/255, 3/255] # pre-set noise standard deviation
        # iter_max = [10, 10, 10, 10, 10] # maximum number of iterations
        sigma    = [50/255, 25/255, 12/255, 6/255] # pre-set noise standard deviation
        iter_max = [20, 20, 20, 20] # maximum number of iterations
        # sigma    = [50/255,25/255] # pre-set noise standard deviation
        # iter_max = [10,10] # maximum number of iterations
        useGPU = True # use GPU
        
        # pre-load the model for FFDNet image denoising
        NUM_IN_FR_EXT = 5 # temporal size of patch
        model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)
        
        # Load saved weights
        state_temp_dict = torch.load('./packages/fastdvdnet/model.pth')
        if useGPU:
            device_ids = [0]
            model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        else:
            # CPU mode: remove the DataParallel wrapper
            state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
        model.load_state_dict(state_temp_dict)
        
        # Sets the model in evaluation mode (e.g. it removes BN)
        model.eval()
        
        begin_time = time.time()
        vgapfastdvdnet_demosaic_bayer_t,psnr_gapfastdvdnet_demosaic_t,ssim_gapfastdvdnet_demosaic_t,psnrall_fastdvdnet_demosaic_t =                 gap_denoise_bayer(meas_bayer_t, mask_bayer, _lambda, 
                                          accelerate, denoiser, iter_max, 
                                          noise_estimate, sigma,
                                          x0_bayer=vgaptv_bayer_t,
                                          X_orig=orig_bayer_t,
                                          model=model)
        end_time = time.time()
        tgapfastdvdnet_demosaic = end_time - begin_time
        print('GAP-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
            denoiser.upper(), mean(psnr_gapfastdvdnet_demosaic_t), mean(ssim_gapfastdvdnet_demosaic_t), tgapfastdvdnet_demosaic))
    
        vgap_fastdvd_color_demosaic_bayer[:,:,iframe*nmask:(iframe+1)*nmask] =  vgapfastdvdnet_demosaic_bayer_t
        psnr_fastdvd_color_demosaic[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(psnr_gapfastdvdnet_demosaic_t,(nmask,1))
        ssim_fastdvd_color_demosaic[iframe*nmask:(iframe+1)*nmask,:] = np.reshape(ssim_gapfastdvdnet_demosaic_t,(nmask,1))
    
        
    savedmatdir = resultsdir + '/savedmat/'
    if not os.path.exists(savedmatdir):
        os.makedirs(savedmatdir)
    # sio.savemat('{}gap{}_{}{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask),
    #             {'vgapdenoise':vgapdenoise},{'psnr_gapdenoise':psnr_gapdenoise})
    sio.savemat('{}gap{}_{}{:d}_sigma{:d}_all7.mat'.format(savedmatdir,denoiser.lower(),datname,nmask,int(sigma[-1]*MAXB)),
                {'vgaptv_bayer':vgaptv_bayer, 
                 'psnr_gaptv':psnr_gaptv,
                 'ssim_gaptv':ssim_gaptv,
                 'tgaptv':tgaptv,
                 'vgapffdnet_bayer':vgapffdgray_bayer, 
                 'psnr_gapffdgray':psnr_gapffdgray,
                 'ssim_gapffdgray':ssim_gapffdgray,
                 't_gapffdgray':tgapffdnet,
                 'vgap_ffd_color_down_bayer':vgap_ffd_color_down_bayer, 
                 'psnr_gapffd_color_down':psnr_gapffd_color_down,
                 'ssim_gapffd_color_down':ssim_gapffd_color_down,
                 'tgapffdnet_color_down':tgapffdnet_color_down,
                 'vgap_ffd_color_demosaic_bayer':vgap_ffd_color_demosaic_bayer, 
                 'psnr_gapffd_color_demosaic':psnr_gapffd_color_demosaic,
                 'ssim_gapffd_color_demosaic':ssim_gapffd_color_demosaic,
                 'tgapffdnet_color_demosaic':tgapffdnet_color_demosaic,
                 'vgap_fastdvd_gray_bayer':vgap_fastdvd_gray_bayer,
                 'psnr_fastdvd_gray':psnr_fastdvd_gray,
                 'ssim_fastdvd_gray':ssim_fastdvd_gray,
                 'tgapfastdvdnet_gray':tgapfastdvdnet_gray,
                 'vgap_fastdvd_color_down_bayer':vgap_fastdvd_color_down_bayer,
                 'psnr_fastdvd_color_down':psnr_fastdvd_color_down,
                 'ssim_fastdvd_color_down':ssim_fastdvd_color_down,
                 'tgapfastdvdnet_down':tgapfastdvdnet_down,
                 'vgap_fastdvd_color_demosaic_bayer':vgap_fastdvd_color_demosaic_bayer,
                 'psnr_fastdvd_color_demosaic':psnr_fastdvd_color_demosaic,
                 'ssim_fastdvd_color_demosaic':ssim_fastdvd_color_demosaic,
                 'tgapfastdvdnet_demosaic':tgapfastdvdnet_demosaic,
                 'orig_real':orig_real,
                 'meas_bayer':meas_bayer})
