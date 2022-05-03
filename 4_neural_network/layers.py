from builtins import range
import numpy as np
import math


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable out. #
    ###########################################################################
    out = x.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.where(x >= 0, dout, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height HH and width WW. 
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    ###########################################################################
    # Extract shapes and constants


    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = 1
    pad = 0
    # Check for parameter sanity
    assert (H + 2 * pad - HH) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
    assert (W + 2 * pad - WW) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride
    # Padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    # Construct output
    out = np.zeros((N, F, H_prime, W_prime))
    # Naive Loops
    #for n in range(N):
    #    for f in range(F):
    #        for j in range(0, H_prime):
    #            for i in range(0, W_prime):
    #                out[n, f, j, i] = (x_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * w[f, :, :, :]).sum()
    #for f in range(F):
    #    for j in range(0, H_prime):
    #        for i in range(0, W_prime):
    #            tmp_w = w[f, :, :, :]
    #            tmp_w = tmp_w[np.newaxis,:]
    #            tmp_w = np.repeat(tmp_w, N, axis=0) 
    #            out[:, f, j, i] = np.sum(np.sum(np.sum(x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * tmp_w, axis=-3), axis=-2), axis=-1)
    for j in range(0, H_prime):
        for i in range(0, W_prime):
           tmp_w = w
           tmp_w = tmp_w[np.newaxis,:]
           tmp_w = np.repeat(tmp_w, N, axis=0)
           tmp_x = x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW] 
           tmp_x = tmp_x[:,np.newaxis]
           tmp_x = np.repeat(tmp_x, F, axis=1)
           out[:,:,j,i] = np.sum(np.sum(np.sum(tmp_x*tmp_w, axis=-1), axis=-1), axis=-1) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = 1
    pad = 0
    # Padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride
    # Construct output
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    # Naive Loops
    #for n in range(N):
    #    for f in range(F):
    #        for j in range(0, H_prime):
    #            for i in range(0, W_prime):
    #                dw[f] += x_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * dout[n, f, j, i]
    #                dx_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] += w[f] * dout[n, f, j, i]
    # Extract dx from dx_pad
    #dx = dx_pad[:, :, pad:pad + H, pad:pad + W]

    #for f in range(F):
    #    for j in range(0, H_prime):
    #        for i in range(0, W_prime):
    #            tmp_dout = dout[:, f, j, i]
    #            tmp_dout = tmp_dout[:,np.newaxis]
    #            tmp_dout = np.repeat(tmp_dout, C, axis=1)
    #            tmp_dout = tmp_dout[:,:,np.newaxis]
    #            tmp_dout = np.repeat(tmp_dout, HH, axis=2)
    #            tmp_dout = tmp_dout[:,:, :, np.newaxis]
    #            tmp_dout = np.repeat(tmp_dout, WW, axis=3)
    #            dw[f] += np.sum(x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * tmp_dout, axis=0)
    #            dx_pad[:,:,j * stride:j * stride + HH, i * stride:i * stride + WW] += w[f]*tmp_dout
    
    for j in range(0, H_prime):
        for i in range(0, W_prime):
            tmp_dout = dout[:, :, j, i] 
            tmp_dout = tmp_dout[:,:,np.newaxis] 
            tmp_dout = np.repeat(tmp_dout, C, axis=2)
            tmp_dout = tmp_dout[:,:,:,np.newaxis]  
            tmp_dout = np.repeat(tmp_dout, HH, axis=3)
            tmp_dout = tmp_dout[:,:,:,:, np.newaxis]
            tmp_dout = np.repeat(tmp_dout, WW, axis=4)
            tmp_x = x_pad[:, :, j * stride:j * stride + HH, i * stride:i * stride + WW]
            tmp_x = tmp_x[:,np.newaxis]
            tmp_x = np.repeat(tmp_x, F, axis=1)
            dw += np.sum(tmp_x * tmp_dout, axis=0)
            tmp_w = w
            tmp_w = tmp_w[:,np.newaxis]
            tmp_w = np.repeat(tmp_w, N, axis=0)
            dx_pad[:,:,j * stride:j * stride + HH, i * stride:i * stride + WW] += np.sum(w*tmp_dout, axis=1)
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here. Output size is given by 
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    (N, C, H, W) = x.shape

    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    S = (N, C, math.floor(H_out), math.floor(W_out))
    out = np.zeros(S)
    #for n in range(N):
    #    for c in range(C):
    #        for x1 in range(math.floor(H_out)):
    #            for y in range(math.floor(W_out)):
    #                out[n, c, x1, y] = np.amax(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W])
    
    for x1 in range(math.floor(H_out)):
        for y in range(math.floor(W_out)):
            out[:, :, x1, y] = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    (x, pool_param) = cache
    (N, C, H, W) = x.shape
    p_H = pool_param.get('pool_height', 3)
    p_W = pool_param.get('pool_width', 3)
    stride = pool_param.get('stride', 1)
    H_out = 1 + (H - p_H) / stride
    W_out = 1 + (W - p_W) / stride

    dx = np.zeros(x.shape)
    #for n in range(N):
    #    for c in range(C):
    #        for x1 in range(math.floor(H_out)):
    #            for y in range(math.floor(W_out)):
    #                max_element = np.amax(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W])
    #                temp = np.zeros(x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W].shape)
    #                temp = (x[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] == max_element)
    #                dx[n, c, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] += dout[n, c, x1, y] * temp
    for x1 in range(math.floor(H_out)):
        for y in range(math.floor(W_out)):
            max_element = np.amax(np.amax(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W], axis=-1), axis=-1)
            max_element = max_element[:,:,np.newaxis]
            max_element = np.repeat(max_element, p_H, axis=2)
            max_element = max_element[:,:,:,np.newaxis]
            max_element = np.repeat(max_element, p_W, axis=3)
            temp = np.zeros(x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W].shape)
            temp = (x[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] == max_element)
            tmp_dout = dout[:,:,x1,y]
            tmp_dout = tmp_dout[:,:,np.newaxis]
            tmp_dout = np.repeat(tmp_dout, p_H, axis=2)
            tmp_dout = tmp_dout[:,:,:,np.newaxis]
            tmp_dout = np.repeat(tmp_dout, p_W, axis=3)
            dx[:, :, x1 * stride: x1 * stride + p_H, y * stride: y * stride + p_W] += tmp_dout * temp
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """

  shifted_logits = x - np.max(x, axis=1, keepdims=True)
  Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
  log_probs = shifted_logits - np.log(Z)
  probs = np.exp(log_probs)
  N = x.shape[0]
  loss = -np.sum(log_probs[np.arange(N), y]) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
