''' Utilities '''
import math
import numpy as np

def A_(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    # # for 3-D measurements only
    # return np.sum(x*Phi, axis=2)  # element-wise product
    # for general N-D measurements
    return np.sum(x*Phi, axis=tuple(range(2,Phi.ndim)))  # element-wise product

def At_(y, Phi):
    '''
    Tanspose of the forward model. 
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    # # for 3-D measurements only
    # return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)
    # for general N-D measurements (original Phi: H x W (x C) x F, y: H x W)
    # [expected] direct broadcasting (Phi: F x C x H x W, y: H x W)
    # [todo] change the dimension order to follow NumPy convention
    # D = Phi.ndim
    # ax = tuple(range(2,D))
    # return np.multiply(np.repeat(np.expand_dims(y,axis=ax),Phi.shape[2:D],axis=ax), Phi) # inefficient considering the memory layout https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
    return (Phi.transpose()*y.transpose()).transpose() # broadcasted by numpy

def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    mse = np.mean( (ref - img) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
