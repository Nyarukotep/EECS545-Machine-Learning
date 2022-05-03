"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""
import numpy as np


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    h_raw = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    next_h = np.tanh(h_raw)

    cache = (Wx, Wh, x, prev_h, h_raw)

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    Wx, Wh, x, prev_h, h_raw = cache

    dh_raw = (1 - np.tanh(h_raw) ** 2) * dnext_h
    dx = np.dot(dh_raw, Wx.T)
    dprev_h = np.dot(dh_raw, Wh.T)
    dWx = np.dot(x.T, dh_raw)
    dWh = np.dot(prev_h.T, dh_raw)
    db = np.sum(dh_raw, axis=0)

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After
    running the RNN forward step, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (T, N, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (T, N, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    T, N, D = x.shape
    H = Wh.shape[0]

    forward_steps_cache = {}
    h = np.zeros([T, N, H])

    for t in range(T):
        if t == 0:
            h[t, :, :], forward_steps_cache[t] = rnn_step_forward(
                x[t, :, :], h0, Wx, Wh, b)
        else:
            h[t, :, :], forward_steps_cache[t] = rnn_step_forward(
                x[t, :, :], h[t-1, :, :], Wx, Wh, b)

    cache = (h0, forward_steps_cache, N, T, D, H)

    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (T, N, H)
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (T, N, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    h0, forward_steps_cache, N, T, D, H = cache
    dx = np.zeros([T, N, D])
    dWx = np.zeros([D, H])
    dWh = np.zeros([H, H])
    db = np.zeros([H])

    dprev_h = 0
    for t in reversed(range(T)):
        dh_ag = dh[t,:, :] + dprev_h
        dx_step, dprev_h, dWx_step, dWh_step, db_step = rnn_step_backward(
            dh_ag, forward_steps_cache[t])
        dWx += dWx_step
        dWh += dWh_step
        db += db_step
        dx[t, :, :] = dx_step

    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning
    each to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    out = W[x, :]
    cache = x, W
    return out, cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW



def temporal_fc_forward(x, w, b):
    """
    Forward pass for a temporal fully-connected layer. The input is a set of
    D-dimensional vectors arranged into a minibatch of N timeseries, each of
    length T. We use an linear function to transform each of those vectors into
    a new vector of dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass.
    ###########################################################################
    out = x @ w + b
    cache = (x, w, b, out)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache



def temporal_fc_backward(dout, cache):
    """
    Backward pass for temporal fully-connected layer.
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    dx, dw, db = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass.
    ###########################################################################
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dw, db


def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at
    some timesteps, since sequences of different length may have been combined
    into a minibatch and padded with NULL tokens. The optional mask argument
    tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the
        range 0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving cross-entropy loss averaged over N samples.
    - dx: Gradient of loss with respect to scores x.
    """
    loss, dx = None, None
    N, T, V = x.shape
    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]
    dx = dx_flat.reshape(N, T, V)

    return loss, dx
