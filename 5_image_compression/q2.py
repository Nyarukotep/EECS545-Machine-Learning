import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.stats import multivariate_normal

def compute_gamma(pi, mu, sigma, data):
  num_data, ndim = data.shape
  num_centroid = mu.shape[0]
  normpdf = np.zeros((num_centroid, num_data))
  for index in range(num_centroid):
    normpdf[index] = multivariate_normal(mu[index], sigma[index]).pdf(data) # normpdf: [Ncentroid x Ndata]
  
  pi_normpdf = pi * normpdf # [Ncentroid x Ndata]
  totals = pi_normpdf.sum(axis=0, keepdims=True)
  gamma = pi_normpdf / totals + 1e-10  # soft assignment (ie., p(zk=1|x)) # [Ncentroid x Ndata]
  return gamma, totals

def train_gmm(train_data, init_pi, init_mu, init_sigma):
  num_data, ndim = train_data.shape
  num_centroid = init_mu.shape[0]

  # init
  pi = np.ones((num_centroid, 1)) / num_centroid # [Ncentroid x 1] 
  mu = init_mu # [Ncentroid x dim] 
  sigma = np.tile(np.identity(ndim), [num_centroid, 1, 1])*1000. # [Ncentroid x dim x dim]

  for iter in range(50):
    # E-step
    gamma, totals = compute_gamma(pi, mu, sigma, train_data) # [Ncentroid x Ndata]
    if np.any(np.isnan(gamma)):
      import pdb; pdb.set_trace()
    Num_points_in_clusters = gamma.sum(axis=1, keepdims=True) # N_k [Ncentroid x Ndata].sum() -> [Ncentroid x 1]
    
    # M-step
    ### \pi
    pi = Num_points_in_clusters / num_data
    ### \mu
    gamma_x_sum = np.matmul(gamma, train_data) # [Ncentroid x Ndata ] x [Ndata x dim] = [Ncentroid x dim]
    mu = gamma_x_sum / Num_points_in_clusters  # [Ncentroid x dim]
    if np.any(np.isnan(mu)):
      import pdb; pdb.set_trace()

    print('iter={:2d}: log-likelihood={:6.1f}\\\\'.format(iter, get_likelihood(totals)))
    """  # slow-version
    ### \Sigma
    for index in range(num_centroid):
      sum_mat = 0.
      for data_ind, data in enumerate(train_data):
        x_zero_meaned = np.expand_dims(data - mu[index], 1)
        sum_mat += gamma[index, data_ind] * np.matmul(x_zero_meaned, x_zero_meaned.transpose())
      sigma[index] = sum_mat / Num_points_in_clusters[index]  # [dim x dim] / [1] = [dim x dim]"""

    ### \Sigma # fast-version
    for index in range(num_centroid):
      x_scaled_zero_meaned = (train_data - mu[[index], :]) * np.expand_dims(np.sqrt(gamma[index]), 1) # [Ndata x dim] * [Ndata x 1]
      covar = np.cov( x_scaled_zero_meaned.transpose() ) # [dim x dim]
      sigma[index] = covar / Num_points_in_clusters[index] * num_data  # [dim x dim] / [1] = [dim x dim]
      if np.any(np.isnan(sigma[index])):
        import pdb; pdb.set_trace()
  states = {
      'pi': pi,
      'mu': mu,
      'sigma': sigma,
  }
  return states

def get_likelihood(totals):
  likelihood = []
  sample_likelihoods = np.log(totals)
  return np.sum(sample_likelihoods)

def test_gmm(states, test_data):
  pi = states['pi']
  mu = states['mu']
  sigma = states['sigma']
  ##### TODO: Implement here!! #####
  num_centroids = mu.shape[0]
  gamma, _ = compute_gamma(pi, mu, sigma, test_data) # [Ncentroid x Ndata]
  assignment = np.argmax(gamma, axis=0)
  compressed_data = test_data.copy()
  
  """
  for index in range(num_centroids):
    mask = assignment == index
    compressed_data[mask, :] = mu[index] # hard-assignment"""

  compressed_data = np.matmul(gamma.transpose(), mu) # [Ndata x Ncentroid] x [Ncentroid x dim] # soft-assignment
  plt.imshow(compressed_data.reshape((512, 512, 3)).astype(np.uint8))
  plt.savefig('q2_gmm.png')
  ##### TODO: Implement here!! #####
  
  result = {
      'pixel-error': calculate_error(test_data, compressed_data),
  }
  return result

### DO NOT CHANGE ###
def calculate_error(data, compressed_data):
  assert data.shape == compressed_data.shape
  error = np.sqrt(np.mean(np.power(data - compressed_data, 2)))
  return error
### DO NOT CHANGE ###

# Load data
img_small = np.array(imageio.imread('q12data/mandrill-small.tiff')) # 128 x 128 x 3
img_large = np.array(imageio.imread('q12data/mandrill-large.tiff')) # 512 x 512 x 3

ndim = img_small.shape[-1]

train_data = img_small.reshape(-1, 3).astype(float)
test_data = img_large.reshape(-1, 3).astype(float)

# GMM
num_centroid = 5
initial_mu_indices = [16041, 15086, 15419,  3018,  5894]
init_pi = np.ones((num_centroid, 1)) / num_centroid
init_mu = train_data[initial_mu_indices, :]
init_sigma = np.tile(np.identity(ndim), [num_centroid, 1, 1])*1000.

gmm_states = train_gmm(train_data, init_pi, init_mu, init_sigma)
result_gmm = test_gmm(gmm_states, test_data)

def pretty_print(arr):
  if arr.ndim == 2:
    W, H = arr.shape
    for i in range(W):
      string_line = '$['
      for j, val in enumerate(arr[i]):
        string_line += '%.1f'%(val)
        if j < H - 1:
          string_line += ', '
      string_line += ']$\\\\'
      print(string_line)

np.set_printoptions(precision=1)
print('mu=')
print(gmm_states['mu'])
#pretty_print(gmm_states['mu'])
plt.figure()
plt.imshow(test_data.reshape((512, 512, 3)).astype(np.uint8))
plt.savefig('q2_ground_truth.png')

print('\Sigma=')
print(gmm_states['sigma'])

print('GMM result=', result_gmm)

