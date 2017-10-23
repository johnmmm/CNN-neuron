import numpy as np
import scipy.signal
from datetime import datetime

def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    
    batch, _, h_in, w_in = input.shape
    c_out, c_in, _, _ = W.shape

    h_out = h_in + 2 * pad - kernel_size + 1
    w_out = w_in + 2 * pad - kernel_size + 1
    
    input_pad = np.lib.pad(input, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant', constant_values=0)

    output = np.zeros(batch*c_out*h_out*w_out).reshape(batch, c_out, h_out, w_out)
    for i in range(0, batch):
        for j in range(0, c_out):
            for k in range(0, c_in):
                output[i, j, :, :] += scipy.signal.convolve2d(input_pad[i, k, :, :], np.rot90(W[j, k, :, :], 2), mode='valid')+b[j]

    return output

def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    
    batch, c_in, h_in, w_in = input.shape
    _, c_out, h_out, w_out = grad_output.shape

    grad_input = np.zeros(batch*c_in*h_in*w_in*c_out).reshape(batch, c_in, c_out, h_in, w_in)
    grad_W = np.zeros(c_out*c_in*kernel_size*kernel_size*batch).reshape(c_out, c_in, batch, kernel_size, kernel_size)
    grad_b = np.zeros(c_out)
    
    grad_output = grad_output[:, :, 1:h_out-1, 1:w_out-1]

    for i in range(0, batch):
        for j in range(0, c_in):
            for k in range(0, c_out):
                grad_input[i, j, :, :] = scipy.signal.convolve2d(grad_output[i, k, :, :], W[k, j, :, :], mode='full')
                # temp = scipy.signal.convolve2d(grad_output[i, k, :, :], W[k, j, :, :], mode='full')
                # grad_input[i, j, k, :, :] = temp[pad:h_in+pad, pad:w_in+pad]
                #need to change!
    grad_input = grad_input.sum(axis=2)

    for i in range(0, batch):
        for j in range(0, c_in):
            for k in range(0, c_out):  
                grad_W[k, j, i, :, :] = scipy.signal.convolve2d(input[i, j, :, :], np.rot90(grad_output[i, k, :, :], 2), mode='valid')
    grad_W = grad_W.sum(axis=2)
    grad_b = np.sum(grad_output, axis=(0,2,3)) #ok

    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    
    batch, c_in, h_in, w_in = input.shape
    h_out = (int)((h_in + 2 * pad) / kernel_size)
    w_out = (int)((w_in + 2 * pad) / kernel_size)

    input_tem = input.reshape(batch, c_in, h_out, kernel_size, w_out, kernel_size)
    output = np.mean(input_tem, axis=(3,5))

    return output


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''

    batch, c_in, h_in, w_in = input.shape
    h_out = (h_in + 2 * pad) / kernel_size
    w_out = (w_in + 2 * pad) / kernel_size

    poolingsize = kernel_size**2
    temp = grad_output.repeat(kernel_size, axis=2).repeat(kernel_size, axis=3)
    temp = temp / poolingsize
    grad_input = temp[:, :, pad:pad+h_in, pad:pad+w_in]
    return grad_input
