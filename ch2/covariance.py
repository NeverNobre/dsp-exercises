import numpy as np

# Generate data
mean = np.array([30, -20])
covariance = np.array([[10, .99], [.99, 40]])
data = np.random.multivariate_normal(mean, covariance, 10000)

# Remove means (center the data)
centered_data = data - np.mean(data, axis=0)

# --- Gram-Schmidt Orthonormalization ---
def gram_schmidt(v1, v2):
    # First basis vector (normalized)
    u1 = v1 / np.linalg.norm(v1)
    # Second basis vector (orthogonalized and normalized)
    u2 = v2 - np.dot(v2, u1) * u1
    u2 = u2 / np.linalg.norm(u2)
    return u1, u2

# Initial arbitrary basis (standard basis)
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Compute orthonormal basis
u1, u2 = gram_schmidt(v1, v2)

# Gram-Schmidt transform matrix (columns are u1, u2)
gs_transform = np.column_stack((u1, u2))

# Project data using Gram-Schmidt transform
gs_projected = centered_data @ gs_transform

# Compute variances of the projected data
gs_variances = np.var(gs_projected, axis=0)

# --- PCA Transform ---
# Compute covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# PCA transform matrix (columns are sorted eigenvectors)
pca_transform = eigenvectors

# Project data using PCA transform
pca_projected = centered_data @ pca_transform

# Compute variances of the projected data
pca_variances = np.var(pca_projected, axis=0)

# --- Results ---
print(f"PCA transform: {pca_transform}")
print(f"GS transform: {gs_transform}")
print(f"Gram-Schmidt variances: {gs_variances}")
print(f"PCA variances: {pca_variances}")