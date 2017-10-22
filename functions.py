import numpy as np
import scipy.signal

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
    output = np.random.randn(c_out, batch, h_out, w_out)

    # input = input.transpose(0,2,1,3)
    # y_p = np.sum(input, axis=(0,2,3))
    # print(h_in)
    # print(w_in)
    input_pad = np.lib.pad(input, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant', constant_values=0)
    #np.lib.pad(a, (2,3), 'constant', constant_values=(4, 6))
    y_p = np.sum(input_pad, axis=1)

    # for i in range(0, batch):
    #     for j in range(0, c_out):
    #         for k in range(0, c_in):
    #             output[i][j] += scipy.signal.convolve2d(y_p[i], np.rot90(W[j, k, :, :], 2), mode='valid') + b[j]

    W_3 = W.reshape(c_out, c_in, 1, kernel_size, kernel_size)

    for j in range(0, c_out):
        for k in range(0, c_in):
            output[j, :, :, :] += scipy.signal.fftconvolve(y_p, np.rot90(W_3[j, k, :, :, :], 2), mode='valid') + b[j]
    output = output.transpose(1,0,2,3)
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
    grad_input = np.zeros(batch*c_in*h_in*w_in).reshape(batch, c_in, h_in, w_in)
    grad_W = np.zeros(c_out*c_in*kernel_size*kernel_size).reshape(c_out, c_in, kernel_size, kernel_size)
    grad_b = np.zeros(c_out)
    grad_output = grad_output[:, :, 1:h_out-1, 1:w_out-1]
    #print(grad_output.shape)

    for i in range(0, batch):
        for j in range(0, c_in):
            for k in range(0, c_out):
                grad_input[i, j, :, :] += scipy.signal.convolve2d(grad_output[i, k, :, :], W[k, j, :, :], mode='full')
                grad_W[k, j, :, :] += scipy.signal.convolve2d(input[i, j, :, :], np.rot90(grad_output[i, k, :, :], 2), mode='valid')
    grad_b = np.sum(grad_output, axis=(0,2,3))

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
    h_pool = (int)((h_in + 2 * pad)/h_out)
    w_pool = (int)((w_in + 2 * pad)/w_out)

    output = np.random.randn(batch, c_in, h_out, w_out)
    temp = np.random.randn(h_pool, w_pool)
    # print(h_out)
    # print(w_out)
    # print(h_pool)
    # print(w_pool)
    for i in range(0, batch):
        for j in range(0, c_in):
            for k in range(0, h_out):
                for l in range(0, w_out):
                    startX = k * h_pool
                    startY = l * w_pool
                    #print(input[i][j][startX:startX+h_pool][startY:startY+w_pool])
                    temp = input[i, j, startX:startX+h_pool, startY:startY+w_pool]
                    output[i, j, k, l] = np.mean(temp)
    # print('an?')
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
    h_pool = (h_in + 2 * pad)/h_out
    w_pool = (w_in + 2 * pad)/w_out
    poolingsize = h_pool * w_pool
    grad_input = grad_output.repeat(h_pool, axis=2).repeat(w_pool, axis=3)
    grad_input = grad_input / poolingsize
    # print(len(grad_input))
    return grad_input
