import numpy as np
from PIL import Image
from math import sqrt
from utils.common_utils import *
from copy import deepcopy
from scipy import fftpack


def _psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf


def _otf2psf(otf, psf_size, real=True):
    # calculate psf from otf with size <= otf size
    
    if otf.any(): # if any otf element is non-zero
        # calculate psf     
        psf = np.fft.ifftn(otf)
        # this condition depends on psf size    
        num_small = np.log2(otf.shape[0])*4*np.spacing(1)    
        if np.max(abs(psf.imag))/np.max(abs(psf)) <= num_small:
            psf = psf.real 
        
        # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0]/2)), axis=0)    
        psf = np.roll(psf, int(np.floor(psf_size[1]/2)), axis=1) 
        
        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else: # if all otf elements are zero
        psf = np.zeros(psf_size)
    if real:
        psf = np.real(psf)
    return psf


def psf2otf(psf, outSize=None):
    psf = np.array(psf)
    if outSize == None:
        return _psf2otf(psf, list(psf.shape))
    else:
        return _psf2otf(psf, outSize)
    
    
def otf2psf(otf, outSize=None, real=True):
    otf = np.array(otf)
    if outSize == None:
        return _otf2psf(otf, list(otf.shape), real)
    else:
        return _otf2psf(otf, outSize, real)
    

def solve_min_laplacian(boundary_image):
    (H, W) = np.shape(boundary_image)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(2, H)-1
    k = np.arange(2, W)-1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4*boundary_image[np.ix_(j, k)] + boundary_image[np.ix_(j, k+1)] + boundary_image[np.ix_(j, k-1)] + boundary_image[np.ix_(j-1, k)] + boundary_image[np.ix_(j+1, k)]
    
    del(j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del(f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1,1:-1]
    del(f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0)/2
    else:
        tt = fftpack.dst(f2, type=1)/2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0)/2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1)/2) 
    del(f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W-1), np.arange(1, H-1))
    denom = (2*np.cos(np.pi*x/(W-1))-2) + (2*np.cos(np.pi*y/(H-1)) - 2)

    # divide
    f3 = f2sin/denom
    del(f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3*2, type=1, axis=1)/(2*(f3.shape[1]+1))
    else:
        tt = fftpack.idst(f3*2, type=1, axis=0)/(2*(f3.shape[0]+1))
    del(f3)
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1)/(2*(tt.shape[0]+1)))
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1, axis=0)/(2*(tt.shape[1]+1)))
    del(tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct


def opt_fft_size(n):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    #  opt_fft_size.m
    # compute an optimal data length for Fourier transforms
    # written by Sunghyun Cho (sodomau@postech.ac.kr)
    # persistent opt_fft_size_LUT;
    '''

    LUT_size = 2048
    # print("generate opt_fft_size_LUT")
    opt_fft_size_LUT = np.zeros(LUT_size)

    e2 = 1
    while e2 <= LUT_size:
        e3 = e2
        while e3 <= LUT_size:
            e5 = e3
            while e5 <= LUT_size:
                e7 = e5
                while e7 <= LUT_size:
                    if e7 <= LUT_size:
                        opt_fft_size_LUT[e7-1] = e7
                    if e7*11 <= LUT_size:
                        opt_fft_size_LUT[e7*11-1] = e7*11
                    if e7*13 <= LUT_size:
                        opt_fft_size_LUT[e7*13-1] = e7*13
                    e7 = e7 * 7
                e5 = e5 * 5
            e3 = e3 * 3
        e2 = e2 * 2

    nn = 0
    for i in range(LUT_size, 0, -1):
        if opt_fft_size_LUT[i-1] != 0:
            nn = i-1
        else:
            opt_fft_size_LUT[i-1] = nn+1

    m = np.zeros(len(n))
    for c in range(len(n)):
        nn = n[c]
        if nn <= LUT_size:
            m[c] = opt_fft_size_LUT[nn-1]
        else:
            m[c] = -1
    return m


def wrap_boundary_liu(img, img_size):

    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if img.ndim == 2:
        ret = wrap_boundary(img, img_size)
    elif img.ndim == 3:
        ret = [wrap_boundary(img[:, :, i], img_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary(img, img_size):

    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(img)
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = img[:, :]

    r_A = np.zeros((alpha*2+H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w)/(H_w-1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1-a)*r_A[alpha-1, 0] + a*r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1-a)*r_A[alpha-1, -1] + a*r_A[-alpha, -1]

    r_B = np.zeros((H, alpha*2+W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w)/(W_w-1)
    r_B[0, alpha:-alpha] = (1-a)*r_B[0, alpha-1] + a*r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1-a)*r_B[-1, alpha-1] + a*r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha-1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha-1:])
        r_A[alpha-1:, :] = A2
        r_B[:, alpha-1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha-1:-alpha+1, :])
        r_A[alpha-1:-alpha+1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha-1:-alpha+1])
        r_B[:, alpha-1:-alpha+1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha*2+H_w, alpha*2+W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha-1:, alpha-1:])
        r_C[alpha-1:, alpha-1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha-1:-alpha+1, alpha-1:-alpha+1])
        r_C[alpha-1:-alpha+1, alpha-1:-alpha+1] = C2
    C = r_C
    # return C
    A = A[alpha-1:-alpha-1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret


def circular_gradient(img):
    y = np.diff(img, axis=1)
    z = np.array([img[...,0] - img[...,-1]]).reshape((-1, 1))
    w = np.concatenate([y, z], 1) #horizontal
    a = np.diff(img, axis=0)
    b = np.array([img[0, ...] - img[-1, ...]])
    c = np.concatenate([a, b], 0) #vertical
    return c, w #vertical, horizontal


def valid_gradient_h(img):
    grad = np.zeros_like(img)
    [h, w] = img.shape
    assert h >= 2 and w >= 2
    g = -np.diff(img[1:, :])
    grad[:h-1, :w-1] = g
    return grad


def valid_gradient_v(img):
    grad = np.zeros_like(img)
    [h, w] = img.shape
    assert h >= 2 and w >= 2
    g = -np.diff(img[:,1:],axis=0)
    grad[:h-1, :w-1] = g
    return grad


def valid_gradient(img):
    v = valid_gradient_v(img)
    h = valid_gradient_h(img)
    return v, h


def getlatenthv(S, threshold=None):
    if not threshold:
        Sv, Sh = circular_gradient(S)
    else:
        Sv, Sh = None, None
    return Sv, Sh


def _deblur_adm_aniso(B, psf, lambda_tv=0.003):
    beta = 1/lambda_tv
    beta_rate = 2*sqrt(2)
    beta_min = 0.001
    I = deepcopy(B)
    np_to_im(B, 'B.png')
    otfk = psf2otf(psf, B.shape)
    normin1 = np.conj(otfk) * np.fft.fft2(B)
    denormin1 = np.abs(otfk)**2
    initial = np.real(np.fft.ifft2(np.divide(normin1, denormin1)))
    np_to_im(initial, 'initial.png')
    dh = [[1, -1]];
    dv = [[1], [-1]];
    DH = psf2otf(dh, B.shape)
    DV = psf2otf(dv, B.shape)
    denormin2 = np.abs(DH) ** 2 + np.abs(DV) ** 2
    [Iy, Ix] = circular_gradient(I)
    while beta > beta_min:
        gamma = 1/(2*beta);
        denormin = denormin1 + gamma * denormin2;
        Wx = np.maximum(np.abs(Ix) - beta*lambda_tv, 0)*np.sign(Ix);
        Wy = np.maximum(np.abs(Iy) - beta*lambda_tv, 0)*np.sign(Iy);
        tmp1 = np.array([Wx[...,-1] - Wx[...,0]]).reshape((-1, 1))
        tmp2 = -np.diff(Wx, axis=1)
        W = np.concatenate([tmp1, tmp2], 1)
        tmp3 = np.array([Wy[-1, ...] - Wy[0, ...]])
        tmp4 = -np.diff(Wy, axis=0)
        W += np.concatenate([tmp3, tmp4], 0)
        normin = normin1 + gamma*np.fft.fft2(W)
        fftout = np.divide(normin, denormin)
        I = np.real(np.fft.ifft2(fftout))
        [Iy, Ix] = circular_gradient(I)
        np_to_im(I, 'I_%s.png'%beta)
        beta /= 2
    I[I < 0.0] = 0.0
    I[I > 1.0] = 1.0
    return I


def _deblur_l0(img, psf, lambda_l0=4e-3, kappa=2.0, betamax=1e5):
    shape = np.array(img.shape) + np.array(psf.shape) - 1
    fftsize = opt_fft_size(shape)
    Blurry = deepcopy(img)
    Blurry = wrap_boundary_liu(Blurry, fftsize)
    np_to_im(Blurry, 'Blurry.png')
    beta = 2 * lambda_l0
    PSF = psf2otf(psf, Blurry.shape)
    den_ker = np.abs(PSF) ** 2
    normin1 = np.conj(PSF) * np.fft.fft2(Blurry)
    dh = [[1, -1]];
    dv = [[1], [-1]];
    DH = psf2otf(dh, Blurry.shape)
    DV = psf2otf(dv, Blurry.shape)
    denormin1 = np.abs(DH) ** 2 + np.abs(DV) ** 2
    S = deepcopy(Blurry)
    while beta < betamax:
        denormin = den_ker + beta*denormin1;  
        v, h = circular_gradient(S)
        sumsquare = h**2 + v**2
        t = sumsquare < (lambda_l0 / beta)
        v[t] = 0
        h[t] = 0
        tmp1 = np.array([h[...,-1] - h[...,0]]).reshape((-1, 1))
        tmp2 = -np.diff(h, axis=1)
        W = np.concatenate([tmp1, tmp2], 1)
        tmp3 = np.array([v[0, ...] - v[-1, ...]])
        tmp4 = -np.diff(v, axis=0)
        W += np.concatenate([tmp3, tmp4], 0)
        FS = np.divide((normin1 + beta*np.fft.fft2(W)),denormin)
        S = np.real(np.fft.ifft2(FS))
        np_to_im2(S, 'S%s.png'%beta)
        beta *= kappa
    S = S[:img.shape[0], :img.shape[1]]
    return S
        
            

def deblur(img, exemplar, iters=5, lambda_l0=4e-3, betamax=1e5, kernelsize=[55, 55], lambda_kernel=2.0):
    assert img.shape == exemplar.shape
    assert len(kernelsize) == 2 and kernelsize[0] % 2 == 1 and kernelsize[1] % 2 == 1
    img2 = deepcopy(img)
    shape = np.array(img.shape) + np.array(kernelsize) - 1
    fftsize = opt_fft_size(shape)
    img2 = wrap_boundary_liu(img2, fftsize)
    dh = [[1, -1]];
    dv = [[1], [-1]];
    DH = psf2otf(dh, img2.shape)
    DV = psf2otf(dv, img2.shape)
    gradv, gradh = valid_gradient(img)
    fftgradv = np.fft.fft2(gradv)
    fftgradh = np.fft.fft2(gradh)
    denormin1 = np.abs(DH) ** 2 + np.abs(DV) ** 2
    for i in range(iters):
        '''
        Estimate blur kernel
        '''
        if i == 0:
            I = deepcopy(exemplar)
        Iv, Ih = valid_gradient(I)
        Iv = Iv/np.linalg.norm(Iv, 2)
        Ih = Ih/np.linalg.norm(Ih, 2)
        fftIv = np.fft.fft2(Iv)
        fftIh = np.fft.fft2(Ih)
        normin = np.conj(fftIv) * fftgradv + np.conj(fftIh) * fftgradh
        denormin = np.abs(fftIv) ** 2 + np.abs(fftIh) ** 2 + lambda_kernel
        psf = otf2psf(np.divide(normin, denormin), kernelsize)
        psf = centralize_refine(psf)
        PSF = psf2otf(psf, img2.shape)
        beta = 2 * lambda_l0
        den_ker = np.abs(PSF) ** 2
        normin1 = np.conj(PSF) * np.fft.fft2(img2)
        S = deepcopy(img2)
        while beta < betamax:
            denormin = den_ker + beta*denormin1;  
            v, h = circular_gradient(S)
            sumsquare = h**2 + v**2
            t = sumsquare < (lambda_l0 / beta)
            v[t] = 0
            h[t] = 0
            '''
            Estimate S
            '''
            tmp1 = np.array([h[...,-1] - h[...,0]]).reshape((-1, 1))
            tmp2 = -np.diff(h, axis=1)
            W = np.concatenate([tmp1, tmp2], 1)
            tmp3 = np.array([v[0, ...] - v[-1, ...]])
            tmp4 = -np.diff(v, axis=0)
            W += np.concatenate([tmp3, tmp4], 0)
            FS = np.divide((normin1 + beta*np.fft.fft2(W)),denormin)
            S = np.real(np.fft.ifft2(FS))
            beta *= 2
        S = S[:img.shape[0], :img.shape[1]] 
        I = deepcopy(S)
    np_vis(psf, vispath="psf.png", path="psf.txt")
    final = _deblur_l0(255*img, psf, 2*lambda_l0)
    im = np_to_im(final, 'final.png')
    return S, psf
       
        
def calldeblur(imgpath, exemplarpath, iters=5, lambda_l0=2e-3, betamax=1e5, kernelsize=[55, 55], weight_kernel=2.0):
    img = np.array(Image.open(imgpath).convert('L')).astype(np.float32)/255.0
    exemplar = Image.open(exemplarpath).convert('L')
    if exemplar.size[0] != img.shape[0] or exemplar.size[1] != img.shape[1]:
        exemplar = exemplar.resize(img.shape)
    exemplar = np.array(exemplar).astype(np.float32)/255.0
    deblur(img, exemplar, iters, lambda_l0, betamax, kernelsize, weight_kernel)
    
    
    
if __name__ == '__main__':
    calldeblur('1_gt.png', '1_result.png', kernelsize=[45, 45], iters=2)
    