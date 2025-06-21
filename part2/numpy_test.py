import numpy as np

height = 2
width = 2
input_width = 3
i = 1
j = 1

y_indices = np.arange(i, i+height)[:, None]
x_indices = np.arange(j, j+width)[None, :]
linear_indices = y_indices * input_width + x_indices
print(linear_indices.flatten())
