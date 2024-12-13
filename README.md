#Code for Yun Data Science Test


import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

# Create a sample DataFrame
np.random.seed(42)  # For reproducibility
data = {'x': np.random.uniform(0, 10, 10), 'y': np.random.uniform(0, 10, 10)}
df = pd.DataFrame(data)

# Compute the pairwise Euclidean distance matrix
dist_matrix = distance_matrix(df[['x', 'y']], df[['x', 'y']])

# Group points within a distance of 0.5
within_threshold = dist_matrix <= 0.5

# Use connected components logic to assign groups
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

graph = csr_matrix(within_threshold)
n_components, labels = connected_components(csgraph=graph, directed=False)

# Add group labels to the DataFrame
df['group'] = labels

# Display the resulting DataFrame
print(df)
