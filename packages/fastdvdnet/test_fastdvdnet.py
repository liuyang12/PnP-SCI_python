#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
from .models import FastDVDnet
from .fastdvdnet import fastdvdnet_seqdenoise, denoise_seq_fastdvdnet
from .utils import batch_psnr, init_logger_test, \
				variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger

NUM_IN_FR_EXT = 5 # temporal size of patch
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.png' # output images format

def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
	"""Saves the denoised and noisy sequences under save_dir
	"""
	seq_len = seqnoisy.size()[0]
	for idx in range(seq_len):
		# Build Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir,\
						('n{}_{}').format(sigmaval, idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{}').format(sigmaval, idx) + fext)
		else:
			out_name = os.path.join(save_dir,\
					('n{}_FastDVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

		# Save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name, noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)

def test_fastdvdnet(**args):
	r"""Denoises all sequences present in a given folder. Sequences must be stored as numbered
	image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

	Inputs:
		args (dict) fields:
			"model_file": path to model
			"test_path": path to sequence to denoise
			"suffix": suffix to add to output name
			"max_num_fr_per_seq": max number of frames to load per sequence
			"noise_sigma": noise level used on test set
			"dont_save_results: if True, don't save output images
			"no_gpu": if True, run model on CPU
			"save_path": where to save outputs as png
			"gray": if True, perform denoising of grayscale images instead of RGB
	"""
	# Start time
	start_time = time.time()

	# If save_path does not exist, create it
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	logger = init_logger_test(args['save_path'])

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# Create models
	print('Loading models ...')
	model_temp = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

	# Load saved weights
	state_temp_dict = torch.load(args['model_file'])
	if args['cuda']:
		device_ids = [0]
		model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
	else:
		# CPU mode: remove the DataParallel wrapper
		state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
	model_temp.load_state_dict(state_temp_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model_temp.eval()

	with torch.no_grad():
		# process data
		seq, _, _ = open_sequence(args['test_path'],\
									args['gray'],\
									expand_if_needed=False,\
									max_num_fr=args['max_num_fr_per_seq'])
		seq = torch.from_numpy(seq).to(device)
		# seq_time = time.time() # original starter

		if args['noisy_input']:
			seqn, _, _ = open_sequence('%s_sigma%d'%(args['test_path'],args['noise_sigma']*255),\
									args['gray'],\
									expand_if_needed=False,\
									max_num_fr=args['max_num_fr_per_seq'])
			seqn = torch.from_numpy(seqn).to(device)
		else:
			# Add noise
			noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device)
			seqn = seq + noise

		noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

		seq_time = time.time() # modified starter
		denframes = denoise_seq_fastdvdnet(seq=seqn,\
										noise_std=noisestd,\
										windsize=NUM_IN_FR_EXT,\
										model=model_temp)

	# Compute PSNR and log it
	stop_time = time.time()
	psnr = batch_psnr(denframes, seq, 1.)
	psnr_noisy = batch_psnr(seqn.squeeze(), seq, 1.)
	loadtime = (seq_time - start_time)
	runtime = (stop_time - seq_time)
	seq_length = seq.size()[0]
	logger.info("Finished denoising {}".format(args['test_path']))
	logger.info("\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s".\
				 format(seq_length, runtime, loadtime))
	logger.info("\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))

	# Save outputs
	if not args['dont_save_results']:
		if args['gray']:
			denframes = torch.mean(denframes,dim=-3)
		# Save sequence
		save_out_seq(seqn, denframes, args['save_path'], \
					   int(args['noise_sigma']*255), args['suffix'], args['save_noisy'])

	# close logger
	close_logger(logger)

def fastdvdnet_denoiser(vnoisy, sigma, model=None, useGPU=True, gray=False):
	r"""Denoise an input video (H x W x F x C for color video, and H x W x F for
	     grayscale video) with FastDVDnet
	"""
	# start_time = time.time()
	nColor = 1 if gray else 3 # number of color channels (3 - RGB color, 1 - grayscale)
	# Sets data type according to CPU or GPU modes
	if useGPU:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	if model is None:
		model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT, num_color_channels=nColor)

		# Load saved weights
		if gray:
			state_temp_dict = torch.load('model_gray.pth') # [pre-trained] model for grayscale videos
		else:
			state_temp_dict = torch.load('model.pth') # [pre-trained] model for color videos
		if useGPU:
			device_ids = [0]
			model = nn.DataParallel(model, device_ids=device_ids).cuda()
		else:
			# CPU mode: remove the DataParallel wrapper
			state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
		model.load_state_dict(state_temp_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model.eval()

	with torch.no_grad():
		# vnoisy = vnoisy.transpose((2,3,0,1)) # [do it in torch] from H x W x F x C to F x C x H x W
		vnoisy = torch.from_numpy(vnoisy).type('torch.FloatTensor').to(device)
		noisestd = torch.FloatTensor([sigma]).to(device)

		if gray:
			vnoisy = vnoisy.unsqueeze(3) # unsqueeze the color dimension - [H,W,F] to [H,W,F,C=1]
		vnoisy = vnoisy.permute(2, 3, 0, 1) # from H x W x F x C to F x C x H x W 
		# print(vnoisy.finfo, noisestd.finfo)

		# print(torch.max(vnoisy),torch.min(vnoisy))
		# vnoisy = torch.clamp(vnoisy,0.,1.)
		outv = fastdvdnet_seqdenoise( seq=vnoisy,\
									  noise_std=noisestd,\
									  windsize=NUM_IN_FR_EXT,\
									  model=model )
		# print(outv.shape)
		# print(torch.max(outv),torch.min(outv))
		outv = outv.permute(2, 3, 0, 1) # back from F x C x H x W to H x W x F x C
		if gray:
			outv = outv.squeeze(3) # squeeze the color dimension - [H,W,F,C=1] to [H,W,F]
		outv = outv.data.cpu().numpy()
		# outv = outv.transpose((2,3,0,1)) # [do it in torch] back from F x C x H x W to H x W x F x C

	# stop_time = time.time()
	# print('    FastDVDnet video denoising eclipsed in {:.3f}s.'.format(stop_time-start_time))
	return outv

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
	parser.add_argument("--model_file", type=str,\
						default="./model.pth", \
						help='path to model of the pretrained denoiser')
	parser.add_argument("--test_path", type=str, default="./data/rgb/Kodak24", \
						help='path to sequence to denoise')
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=25, \
						help='max number of frames to load per sequence')
	parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
	parser.add_argument("--noisy_input", action='store_true', help='with noisy images as input')

	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("--save_path", type=str, default='./results', \
						 help='where to save outputs as png')
	parser.add_argument("--gray", action='store_true',\
						help='perform denoising of grayscale images instead of RGB')

	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FastDVDnet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_fastdvdnet(**vars(argspar))
