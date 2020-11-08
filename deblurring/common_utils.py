import torch
import torch.nn as nn
import torchvision
import sys
import cv2
import numpy as np
from PIL import Image
import PIL
import skimage
import matplotlib.pyplot as plt
import random
from queue import Queue


def darkchannel(tensor, patchsize=35):
    assert patchsize % 2 == 1
    padding = (patchsize - 1)//2
    channel = torch.min(tensor, 1)[0] # 1 is the channel dimension
    dc = -torch.nn.functional.max_pool2d(-channel, kernel_size=patchsize, padding=padding, stride=1)
    return dc

    
def centralize(kernel):
    def clip(x, lower_bound, upper_bound):
        if x > upper_bound:
            return upper_bound
        elif x < lower_bound:
            return lower_bound
        else:
            return x
        
    weight_x = 0.0
    weight_y = 0.0
    assert kernel.shape[0] % 2 == 1
    assert kernel.shape[1] % 2 == 1
    new_kernel = np.zeros_like(kernel)
    center_x = (kernel.shape[0] - 1)//2
    center_y = (kernel.shape[1] - 1)//2
    t = []
    s = 0.0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            #print("kernel_refinement.py --> centralize: kernel[(%s, %s)]=%s"%(i, j, kernel[(i, j)]))
            if kernel[(i, j)] < 1e-7:
                continue
            t.append((i, j))
            weight_x += i * kernel[(i, j)]
            weight_y += j * kernel[(i, j)]
            s += kernel[(i, j)]
    weight_x = int(weight_x/s)
    weight_y = int(weight_y/s)
    #print("kernel_refinement.py --> centralize: weight_x = %s, weight_y = %s"%(weight_x, weight_y))
    for element in t:
        newcoordx = clip(element[0] + center_x - weight_x, 0, kernel.shape[0]-1)
        newcoordy = clip(element[1] + center_y - weight_y, 0, kernel.shape[1]-1)
        #print("kernel_refinement.py --> centralize: newcoordx = %s, newcoordy = %s"%(newcoordx, newcoordy))
        new_kernel[(newcoordx, newcoordy)] = kernel[element]
    return new_kernel
        
    
def refine(kernel, thres=0.1, thres2=0.05, verbose=False):
    maxkernel = np.max(kernel)
    kernel[kernel < (maxkernel * thres2)] = 0.0
    connection_dict = dict()
    q = Queue()
    component = 0
    visit = dict()
    sumdict = dict()
    valid = dict()
    l = dict()
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            visit[(i, j)] = False
            valid[(i, j)] = False
            
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if not visit[(i, j)] and kernel[(i, j)] > 0:
                #print(i, j, "kernel_refinement-->refine i, j", component, "component")
                q.put((i, j))
                visit[(i, j)] = True
                l[component] = set()
                sumdict[component] = 0.0
                while not q.empty():
                    x, y = q.get()
                    l[component].add((x, y))
                    sumdict[component] += kernel[(x, y)]
                    if x > 0 and not visit[(x-1, y)] and kernel[(x-1, y)] > 0:
                        q.put((x-1, y))
                        visit[(x-1, y)] = True
                    if y > 0 and not visit[(x, y-1)] and kernel[(x, y-1)] > 0:
                        q.put((x, y-1))
                        visit[(x, y-1)] = True
                    if x < kernel.shape[0] - 1 and not visit[(x+1, y)] and kernel[(x+1, y)] > 0:
                        q.put((x+1, y))
                        visit[(x+1, y)] = True
                    if y < kernel.shape[1] - 1 and not visit[(x, y+1)] and kernel[(x, y+1)] > 0:
                        q.put((x, y+1))
                        visit[(x, y+1)] = True
                    if x > 0 and y > 0 and not visit[(x-1, y-1)] and kernel[(x-1, y-1)] > 0:
                        q.put((x-1, y-1))
                        visit[(x-1, y-1)] = True
                    if x > 0 and y < kernel.shape[1] - 1 and not visit[(x-1, y+1)] and kernel[(x-1, y+1)] > 0:
                        q.put((x-1, y+1))
                        visit[(x-1, y+1)] = True
                    if x < kernel.shape[0] - 1 and y > 0 and not visit[(x+1, y-1)] and kernel[(x+1, y-1)] > 0:
                        q.put((x+1, y-1))
                        visit[(x+1, y-1)] = True
                    if x < kernel.shape[0] - 1 and y < kernel.shape[1] - 1 and not visit[(x+1, y+1)] and kernel[(x+1, y+1)] > 0:
                        q.put((x+1, y+1))
                        visit[(x+1, y+1)] = True
                #print("kernel_refinement.py --> refine l[%s] = "%component, l[component])
                if sumdict[component] > thres:
                    for element in l[component]:
                        valid[element] = True
                #print(sumdict[component], component)
                component += 1
    if verbose:
        print("[prune_kernel.py] component=%s"%component)
        
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            #print("valid[", (x, y), "] = ", valid[(x, y)])
            if valid[(x, y)] is False:
                kernel[(x, y)] = 0.0
    
    kernel = kernel/np.sum(kernel)
    return kernel


def centralize_refine(kernel, refinement=True):
    kernel[kernel < 0] = 0
    kernel = kernel/np.sum(kernel)
    kernel = centralize(kernel)
    kernel[kernel < 0] = 0
    kernel = kernel/np.sum(kernel)
    if refinement:
        kernel = refine(kernel)
    return kernel


def resize(data, size):
    return skimage.transform.resize(data, tuple(size))


def write2Darray(data, lambda_ex=None, path=None):
    assert len(data.shape) == 2
    def func(x):
        if not lambda_ex:
            return x
        else:
            return lambda_ex(x)
  
    if not path:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                print(func(data[i][j]), end=" ")
            print("")
            
    else:
        with open(path, "w") as f:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(func(data[i][j])) + " ")
                f.write("\n")          
                
                
def np_vis(data, path=None, vispath=None, typevis=2):
    if typevis == 1:
        data_vis = (data - np.min(data))/(np.max(data) - np.min(data))
    elif typevis == 2:
        data_vis = data/np.max(data)
    data_vis = data_vis*255
    data_vis = data_vis.astype(np.uint8)
    if vispath:
        data_vis_im = Image.fromarray(data_vis)
        data_vis_im.save(vispath)
    if path:
        with open(path, "w") as f:
            for i in range(data_vis.shape[0]):
                for j in range(data_vis.shape[1]):
                    f.write("%d "%int(data_vis[i][j]))
                f.write("\n")
    return data_vis
    
    
def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''
    imgsize = img.shape

    new_size = (imgsize[0] - imgsize[0] % d,
                imgsize[1] - imgsize[1] % d)

    bbox = [
            int((imgsize[0] - new_size[0])/2),
            int((imgsize[1] - new_size[1])/2),
            int((imgsize[0] + new_size[0])/2),
            int((imgsize[1] + new_size[1])/2),
    ]

    img_cropped = img[0:new_size[0],0:new_size[1],:]
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters()]
        elif  opt=='down':
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)

    return img

def get_image(path, imsize=-1, L=False):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)
    if L:
        img=img.convert('L')
    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    #torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np, grad=False): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    if not grad:
        ar = np.clip(img_np*255,0,255).astype(np.uint8)
    else:
        ar = np.abs(img_np)
        ar = ar/np.max(ar)
        ar = np.clip(ar*255,0,255).astype(np.uint8)
        #print(ar.shape, "common_utils.py 230 ar.shape")
    if len(img_np.shape) == 2:
        pass
    else:
        if img_np.shape[0] == 1:
            ar = ar[0]
        else:
            ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_pil2(img_np):
    ar = np.clip(img_np,0,255).astype(np.uint8)
    if len(img_np.shape) == 2:
        pass
    else:
        if img_np.shape[0] == 1:
            ar = ar[0]
        else:
            ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np,grad=None):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    f = None
    if grad == 'H':
        f = np.array([[1.0, -1.0]])   
    elif grad == 'W':
        f = np.array([[1.0],[-1.0]])
    if f:
        img_np = cv2.filter2D(img_np, -1, f)
    return torch.from_numpy(img_np)


def im_to_torch(path, imsize=-1, L=False, grad=None, device='cuda'):
    img, img_np = get_image(path, imsize, L)
    img_torch = np_to_torch(img_np, grad=grad)
    if device.find("cuda") >= 0:
        img_torch = img_torch.to(device)
    img_torch = img_torch.unsqueeze(0)
    return img, img_np, img_torch

    
def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def visualize_kernel(kernel, kernel_path=''):
    kernel_np = torch_to_np(kernel)[0] #The dimension of kernel is 1.
    kernel_min = np.min(kernel_np)
    kernel_max = np.max(kernel_np)
    kernel_vis = 255*(kernel_np - kernel_min)/(kernel_max - kernel_min)
    kernel_vis = kernel_vis.astype(np.uint8)
    kernel_vis_im = Image.fromarray(kernel_vis )
    if kernel_path:
        kernel_vis_im.save(kernel_path)
    return kernel_vis_im
        
    

def torch_to_im(img_var, img_path=None, crop=None, grad=None):
    #print(img_var.shape, "img_var.shape")
    im = torch_to_np(img_var)
    #print(im.shape, "im.shape")
    if crop:
        im = im[..., crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
        
    if grad:
        im = np_to_pil(im, True)
    else:
        im = np_to_pil(im, False)
    if img_path:
        im.save(img_path)
    return im


def np_to_im(img_var, img_path=None, crop=None, grad=None):
    if crop:
        img_var = img_var[..., crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    if grad:
        img_var = np_to_pil(img_var, True)
    else:
        img_var = np_to_pil(img_var, False)
    if img_path:
        img_var.save(img_path)
    return img_var
            
    
    
def np_to_im2(img_var, img_path=None, crop=None):
    if crop:
        img_var = img_var[..., crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    img_var = np_to_pil2(img_var)
    if img_path:
        img_var.save(img_path)
    return img_var
    


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=[5000, 10000, 15000], gamma=0.1)  # learning rates
        for j in range(num_iter):
            scheduler.step(j)
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False


def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[wf:wf + wc, hf:hf + hc, :]
            else:
                real[ws::scale, hs::scale, :] = image[wf:wf + wc, hf:hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real


def readimg(path_to_image):
    img = cv2.imread(path_to_image)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(x)

    return img, y, cb, cr


