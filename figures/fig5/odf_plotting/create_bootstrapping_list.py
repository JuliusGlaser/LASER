import numpy as np
import h5py

np.random.seed(42)
# Parameters
num_vectors = 1000
vector_shape = (126, 1)

# Set to track uniqueness
unique_vectors = set()
vectors = []

# Generate unique random binary vectors
while len(vectors) < num_vectors:
    vec = np.random.randint(0, 2, size=vector_shape, dtype=np.int8)
    # Convert to tuple for set uniqueness
    vec_tuple = tuple(vec.flatten())
    if vec_tuple not in unique_vectors:
        unique_vectors.add(vec_tuple)
        vectors.append(vec)

# Convert list to numpy array
vectors = np.array(vectors)

f = h5py.File('bootstrapping_vectors.h5', 'w')
f.create_dataset(name='vectors', data=vectors)
f.close()

print("Shape of final array:", vectors.shape)  # (1000, 126, 1)
