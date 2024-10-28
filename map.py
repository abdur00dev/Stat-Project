import numpy as np

def theta_map(X, y, lambda_):
    
    identity = np.eye(X.shape[1])
    identity[0, 0] = 0  
    theta = np.linalg.inv(X.T @ X + lambda_ * identity) @ X.T @ y
    return theta