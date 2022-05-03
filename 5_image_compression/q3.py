import numpy as np
import matplotlib.pyplot as plt
import time

def validate_PCA(states, train_data):
  from sklearn.decomposition import PCA
  pca = PCA()
  pca.fit(train_data)
  true_matrix = pca.components_.T
  true_ev = pca.explained_variance_
  
  output_matrix = states['transform_matrix']
  error = np.mean(np.abs(np.abs(true_matrix) - np.abs(output_matrix)) / np.abs(true_matrix))
  if error > 0.01:
    print('Matrix is wrong! Error=',error)
  else:
    print('Matrix is correct! Error=', error)

  output_ev = states['eigen_vals']
  error = np.mean(np.abs(true_ev - output_ev) / true_ev)
  if error > 0.01:
    print('Variance is wrong! Error=', error)
  else:
    print('Variance is correct! Error=', error)

def train_PCA(train_data):
  centered_data = train_data - np.mean(train_data, axis=0, keepdims=True)
  n, m = centered_data.shape
  assert np.allclose(centered_data.mean(axis=0), np.zeros(m))
  
  cov = np.cov(centered_data.T)

  eigen_vals, eigen_vecs = np.linalg.eig(cov)
  indices = np.argsort(-eigen_vals)
  matrix = eigen_vecs[:, indices]
  eigenvals = eigen_vals[indices]
  
  states = {
      'transform_matrix': matrix,
      'eigen_vals': eigenvals
  }
  return states

def test_PCA(matrix, train_data):
  x_PCA = np.matmul(train_data, matrix)
  std_PCA = np.std(x_PCA, axis=0)
  plt.plot(std_PCA)

# Load data
start = time.time()
images = np.load('q3data/q3.npy')
num_data = images.shape[0]
train_data = images.reshape(num_data, -1)

states = train_PCA(train_data)
print('training time = %d sec'%(time.time() - start))
plt.title('Eigenvalues')
plt.plot(states['eigen_vals'])
plt.savefig('q3_eigenvalues.png')

plt.figure(1)
plt.title('Eigenvalue Test Variance')
test_PCA(states['transform_matrix'], train_data)

validate_PCA(states, train_data)

plt.figure(2)
plt.title('Variance percent')
cumul_ev = np.cumsum(states['eigen_vals'])
variance_por = cumul_ev / cumul_ev[-1]
plt.plot(variance_por)


np.set_printoptions(precision=1)
print('eigenvalues= ', np.array(states['eigen_vals'][:10]))
states['eigen_vals'][:10]

eigen_vectors = states['transform_matrix']
eigen_vectors[:, :9]

plt.subplot(2, 5, 1)
mean_image = np.mean(images, axis=0)
plt.imshow(mean_image)

for i in range(9):
  plt.subplot(2, 5, i+2)
  eig_image = eigen_vectors[:, i].reshape(48, 42)
  plt.imshow(eig_image)

plt.savefig(f'q3_eigenfaces.png')
index_1 = (variance_por >= 0.95).nonzero()[0][0] + 1
index_2 = (variance_por >= 0.99).nonzero()[0][0] + 1
print(index_1, 1 - index_1 / 48/42)
print(index_2, 1 - index_2 / 48/42)

