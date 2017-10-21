import numpy as np


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
    out_cha, in_cha, _, _ = W.shape

    h_out = h_in + 2 * pad - kernel_size + 1
    w_out = w_in + 2 * pad - kernel_size + 1
    output = np.random.randn(batch, out_cha, h_out, w_out)

    input = np.transpose(0,2,1,3)
    y_p = np.sum(input, axis=(0,2,3))
    for i in range(0, batch):
        for j in range(0, out_cha):
            for k in range(0, in_cha):
                output[i][j] = np.convolve(y_p[i], np.rot90(W[j][k], 2), mode='valid') + b[j]
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
    grad_input = np.random.zero(batch, c_in, h_in, w_in)
    grad_W = np.random.zero(c_out, c_in, kernel_size, kernel_size)
    grad_b = np.random.zero(c_out)

    for i in range(0, batch):
        for j in range(0, c_in):
            for k in range(0, c_out):
                grad_input[i][j] += np.convolve(grad_output[i][k], W[k][j], mode='full')
                grad_W[k][j] += np.convolve(input[i][j], np.rot90(grad_output[i][k], 2), mode='valid')
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
    pass


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
    pass
