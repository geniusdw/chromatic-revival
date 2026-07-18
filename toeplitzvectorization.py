import numpy as np
from scipy.linalg import toeplitz

def toeplitz_1d_convolve(signal, kernel):
    # Determine padding dimensions
    sig_len, ker_len = len(signal), len(kernel)
    out_len = sig_len + ker_len - 1
    
    # Construct the transform matrix rows and columns
    padding_col = np.zeros(out_len)
    padding_col[:ker_len] = kernel
    padding_row = np.zeros(sig_len)
    padding_row[0] = kernel[0]
    
    # Generate the convolution matrix transform
    transform_matrix = toeplitz(padding_col, padding_row)
    
    # Perform standard matrix dot product
    return np.dot(transform_matrix, signal)

# Run transformation
print("Toeplitz Matrix Output:", toeplitz_1d_convolve(np.array([1, 2, 3]), np.array([0, 1, 0.5])))
