
import numpy as np

def theta_mle(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta