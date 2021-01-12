"""
FastDVDnet denoising algorithm

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import torch
import torch.nn.functional as F

def temp_denoise(model, noisyframe, sigma_noise):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%4
	if expanded_h:
		expanded_h = 4-expanded_h
	expanded_w = sh_im[-1]%4
	if expanded_w:
		expanded_w = 4-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')
	# print(noisyframe.shape, sigma_noise.shape)
	# denoise
	# out = torch.clamp(model(noisyframe, sigma_noise), 0., 1.)
	out = model(noisyframe, sigma_noise) # omit value clip

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out

def denoise_seq_fastdvdnet(seq, noise_std, windsize, model):
	r"""Denoises a sequence of frames with FastDVDnet.
	Args:
		seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the added noise
		windsize: size of the temporal patch
		model_temp: instance of the PyTorch model of the temporal denoiser
	Returns:
		denframes: Tensor, [numframes, C, H, W]
	"""
	# init arrays to handle contiguous frames and related patches
	# print(seq.shape)
	numframes, C, H, W = seq.shape
	ctrlfr_idx = int((windsize-1)//2)
	inframes = list()
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	# build noise map from noise std---assuming Gaussian noise
	noise_map = noise_std.expand((1, 1, H, W))

	for fridx in range(numframes):
		# load input frames
		if not inframes:
		# if list not yet created, fill it with temp_patchsz frames
			for idx in range(windsize):
				relidx = abs(idx-ctrlfr_idx) # handle border conditions, mirror padding
				inframes.append(seq[relidx])
		else:
			del inframes[0]
			relidx = min(fridx + ctrlfr_idx, -fridx + 2*(numframes-1)-ctrlfr_idx) # handle border conditions
			inframes.append(seq[relidx])

		inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, windsize*C, H, W)).to(seq.device)

		# append result to output list
		denframes[fridx] = temp_denoise(model, inframes_t, noise_map)

	# free memory up
	del inframes
	del inframes_t
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes


def fastdvdnet_seqdenoise(seq, noise_std, windsize, model):
	r"""Denoising a video sequence with FastDVDnet.
	
	Parameters 
	----------
	seq : array_like [torch.Tensor]
	      Input noisy video sequence data with size of [N, C, H, W] with 
		  N, C, H, and W being the number of frames, number of color channles 
		  (C=3 for color, C=1 for grayscale), height, and width of the video 
		  sequence to be denoised.
	noise_std : array_like [torch.Tensor]
	      Noise standard deviation with size of [H, W].
	windsize : scalar
		  Temporal window size (number of frames as input to the model).
	model : [torch.nn.Module]
		  Pre-trained model for denoising.
	
	Returns
	-------
	seq_denoised : array_like [torch.Tensor]
		  Output denoised video sequence, with the same size as the input, 
		  that is [N, C, H, W].
	"""
	# init arrays to handle contiguous frames and related patches
	# print(seq.shape)
	N, C, H, W = seq.shape
	hw = int((windsize-1)//2) # half window size
	seq_denoised = torch.empty((N, C, H, W)).to(seq.device)
	# input noise map 
	noise_map = noise_std.expand((1, 1, H, W))

	for frameidx in range(N):
		# cicular padding for edge frames in the video sequence
		idx = (torch.tensor(range(frameidx, frameidx+windsize)) - hw) % N # circular padding
		noisy_seq = seq[idx].reshape((1, -1, H, W)) # reshape from [W, C, H, W] to [1, W*C, H, W]
		
		# make sure the width W and height H multiples of 4
		#   pad the width W and height H to multiples of 4
		M = 4 # multipier
		wpad, hpad = W%M, H%M
		if wpad:
			wpad = M-wpad
		if hpad:
			hpad = M-hpad
		pad = (0, wpad, 0, hpad) 
		noisy_seq = F.pad(noisy_seq, pad, mode='reflect')
		noise_map = F.pad(noise_map, pad, mode='reflect')
		
		# apply the denoising model to the input datat
		frame_denoised = model(noisy_seq, noise_map)

		# unpad the results
		if wpad:
			frame_denoised = frame_denoised[:, :, :, :-wpad]
		if hpad:
			frame_denoised = frame_denoised[:, :, :-hpad, :]
		
		seq_denoised[frameidx] = frame_denoised

		# # apply the denoising model to the input datat
		# seq_denoised[frameidx] = model(noisy_seq, noise_map)

	return seq_denoised