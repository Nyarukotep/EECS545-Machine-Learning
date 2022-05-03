import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import pairwise_distances

def train_kmeans(train_data, initial_centroids):
  states = {}
  # Implement your algorithm and return
  num_data = train_data.shape[0]
  num_centroids = initial_centroids.shape[0]
  centroids = initial_centroids
  for iter in range(50):
    # Compute distance and assign
    """
    distance = pairwise_distances(centroids, train_data) # 16 x Ndata
    """
    distance = np.sqrt(
        np.sum(centroids ** 2, axis=1, keepdims=True)
        - (2 * centroids @ train_data.T)
        + np.sum(train_data ** 2, axis=1, keepdims=True).T
    )
    assignment = np.argmin(distance, axis=0)
    pixel_error = np.min(distance, axis=0)
    calc_error = calculate_error(train_data, centroids[assignment])

    # Update centroids
    """
    for index in range(num_centroids):
      mask = assignment == index
      centroids[index] = np.mean(train_data[mask, :], axis=0)
    """
    assignment_indicator = np.eye(num_centroids)[:, assignment]
    centroids = (
        assignment_indicator
        @ train_data
        / np.sum(assignment_indicator, axis=1, keepdims=True)
    )
    print('iter={:2d}: error={:2.2f} calc_error={:2.2f}\\\\'.format(iter,pixel_error.mean(), calc_error))
  
  states = {
      'centroids': centroids
  }
  return states

def test_kmeans(states, test_data):
  result = {}
  ##### TODO: Implement here!! #####
  num_centroids = initial_centroids.shape[0]
  centroids = states['centroids']
  distance = pairwise_distances(centroids, test_data) # 16 x Ndata
  assignment = np.argmin(distance, axis=0)
  compressed_data = test_data.copy()
  
  for index in range(num_centroids):
    mask = assignment == index
    compressed_data[mask, :] = centroids[index]
  ##### TODO: Implement here!! #####
  plt.imshow(compressed_data.reshape((512, 512, 3)).astype(np.uint8))
  plt.savefig('q1_kmeans.png')
  result['pixel-error'] = calculate_error(test_data, compressed_data)
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

# K-means
num_centroid = 16
initial_centroid_indices = [16041, 15086, 15419,  3018,  5894,  6755, 15296, 11460, 
                            10117, 11603, 11095,  6257, 16220, 10027, 11401, 13404]
initial_centroids = train_data[initial_centroid_indices, :]
states = train_kmeans(train_data, initial_centroids)
result_kmeans = test_kmeans(states, test_data)
print('Kmeans result=', result_kmeans)

