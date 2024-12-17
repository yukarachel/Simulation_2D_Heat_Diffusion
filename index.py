#!/usr/bin/env python
# coding: utf-8

# # Simulation of Two-Dimensional Heat Diffusion

# # Using matrix multiplication

# I wrote functions in the file heat_equations, so we need to import heat_equations here. We also need to import all other libraries we used. 

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import time
import jax.numpy as jnp
from jax.experimental import sparse
import jax


# Here, we have the advance_time_matvecmul function which returns the matrix that is 1 timestep after the original matrix. 

# In[16]:


from heat_equation import advance_time_matvecmul, get_A
import inspect
print(inspect.getsource(advance_time_matvecmul))


# In[17]:


def advance_time_matvecmul(A, u, epsilon):
    """Advances the simulation by one timestep, via matrix-vector multiplication
    Args:
        A: The 2d finite difference matrix
        u: N x N grid state at timestep k
        epsilon: stability constant

    Returns:
        N x N Grid state at timestep k+1
    """
    u = u + epsilon * (A @ u.flatten()).reshape((N, N))
    return u


# get_A gets the value of A, which is the matrix that the original matrix, u, will be multiplied by to get the updated matrix.

# In[4]:


def get_A(N):
    """Gets the value of A using diagonals. The matrix A is what the matrix u will be multiplied by to get the 
    second timestep matrix. 
    """
    n = N * N
    diagonals = [-4 * np.ones(n), np.ones(n-1), np.ones(n-1), np.ones(n-N), np.ones(n-N)]
    diagonals[1][(N-1)::N] = 0
    diagonals[2][(N-1)::N] = 0
    A = np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[2], -1) + np.diag(diagonals[3], N) + np.diag(diagonals[4], -N)
    return A


# We then create an array called u0 and fill it with zeros, except for the one unit of heat that is in the center. This is our starting position, and we will do 2700 iterations in total. Our graph will show a visualization every 300 iterations, so that we can see how the heat diffusion is occuring. We have a list called results that will hold the results of each numpy array every 300 iterations, so that we can output them using the list at the end. Right now, we need to make our original array because we will use it for each simulation. It will be called u0.

# In[5]:


# Set the size of the grid (N x N) and the epsilon constant
N = 101
epsilon = 0.2

# The initial condition will be one unit of heat in the middle of the matrix. An array of zeros are made, with just one point being 1.0
u0 = np.zeros((N, N))
u0[int(N/2), int(N/2)] = 1.0
plt.imshow(u0)


# In[6]:


# Initialize list to store results for visualization
results = []

# Allow u to be u0, since we already created the original matrix u0
u = u0

# Initialize the finite difference matrix A. Uses the get_A(N) function to initialize A
A = get_A(N)

# Measure start time
start_time = time.time()

# Run the simulation for 2700 iterations
for i in range(2700):
    u = advance_time_matvecmul(A, u, epsilon)
    if (i + 1) % 300 == 0:
        results.append(u.copy())

# Calculate total time
execution_time = time.time() - start_time
print("Execution time:", execution_time, "seconds")

# Visualize the heat diffusion every 300 iterations
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
count = 0
for i in range(3):
    for j in range(3):
        axs[i][j].imshow(results[count])
        axs[i][j].set_title(f"{(count+1)*300}th iteration")
        count += 1
    
plt.tight_layout()
plt.show()


# That technique accomplished the task very slowly (it took about 70 seconds to execute). Now, we will try a technique that will make it faster. Sparse matrices store only the non-zero values of a matrix, so that it is smaller and more efficient than an array filled with many zeros. We create a sparse matrix by turning our numpy array into a jnp array, then using .fromdense() to turn it into a sparse matrix. 

# # Using sparse matrix in JAX

# In[ ]:


from heat_equation import get_sparse_A
import inspect
print(inspect.getsource(get_sparse_A))


# In[7]:


def get_A_sparse(N):
    """Gets the value of A using sparse in jax.experimental. First, we have to convert the matrix A from a numpy array
    into a jnp matrix using jnp. Then we turn it into a sparse matrix by using .fromdense(). This will speed up the array
    processing by only storing the non-zero values. 
    """
    A = get_A(N)
    jnp_A = jnp.array(A)
    A_sp_matrix = sparse.BCOO.fromdense(jnp_A)
    return A_sp_matrix


# So now, we repeat the same process, but this time, instead of using get_A(), we use get_A_sparse() so that we can utilize the sparse matrix. We also have a new list called results_sp. 

# In[8]:


# Initialize list to store results for visualization
results_sp = []

# Allow u to be u0, since we already created the original matrix u0
u = u0

# Initialize the finite difference matrix A. Uses the get_A(N) function to initialize A
A = get_A_sparse(N)

# Measure start time
start_time = time.time()

# Run the simulation for 2700 iterations
for i in range(2700):
    u = advance_time_matvecmul(A, u, epsilon)
    if (i + 1) % 300 == 0:
        results_sp.append(u.copy())

# Calculate total time
execution_time = time.time() - start_time
print("Execution time:", execution_time, "seconds")

# Visualize the heat diffusion every 300 iterations
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
count = 0
for i in range(3):
    for j in range(3):
        axs[i][j].imshow(results_sp[count])
        axs[i][j].set_title(f"{(count+1)*300}th iteration")
        count += 1
    
plt.tight_layout()
plt.show()


# As you can see, the time that it took to execute was much faster than the first method. It was 3 seconds, but we should make it even faster using numpy calculations. Rather than creating an array that will be multiplied against the original array, we will just take the original array and perform computations on the values to output the folliwing matrix. We utilize np.roll to save 4 different arrays, then use them to compute the updated array. We should use the formula provided. 

# # Using direct operation with numpy

# In[ ]:


from heat_equation import advance_time_numpy
import inspect
print(inspect.getsource(advance_time_numpy))


# In[9]:


def advance_time_numpy(u, epsilon):
    """Advances the matrix by one timestep using numpy arrays to calculate the next matrix 
    after a timestep. We use np.roll to create new arrays that have all their values shifted, then we use them
    to replicate the equation of finding the matrix after 1 time step.
    """
    padded_u = np.pad(u, 1, mode='constant', constant_values=0)
    right_roll = np.roll(padded_u, 1)
    left_roll = np.roll(padded_u, -1)
    top_roll = np.roll(padded_u, -1, axis=0)
    down_roll = np.roll(padded_u, 1, axis=0)
    updated_u = padded_u + epsilon*(right_roll + left_roll + top_roll + down_roll - (4*padded_u))
    updated_u_sliced = updated_u[1:-1, 1:-1]
    
    return updated_u_sliced


# So now, we create an original array like before, but instead of creating an array called A, we update the matrix using computations in numpy itself.

# In[10]:


# Initialize list to store results for visualization
results_np = []

# Allow np_u to be u0, since we already created the original matrix u0
np_u = u0

# Measure start time
start_time = time.time()

# Run the simulation for 2700 iterations
for i in range(2700):
    np_u = advance_time_numpy(np_u, epsilon)
    if (i + 1) % 300 == 0:
        results_np.append(np_u.copy())

# Calculate total time
execution_time = time.time() - start_time
print("Execution time:", execution_time, "seconds")

# Visualize the heat diffusion every 300 iterations
fig,axs = plt.subplots(3, 3, figsize=(12, 12))
count = 0
for i in range(3):
    for j in range(3):
        axs[i][j].imshow(results_np[count])
        axs[i][j].set_title(f"{(count+1)*300}th iteration")
        count += 1
    
plt.tight_layout()
plt.show()


# This gave us a pretty good execution time of 0.25 seconds, but we can actually make it even faster using jit compilation from JAX. We just use the same function from advance_time_numpy(), but we change the np.roll()s to jnp.roll()s, so that the compilation happens jit, which means just in time. It is faster because it compiles just before it is shown. 

# # Using JAX

# In[ ]:


from heat_equation import advance_time_jax
import inspect
print(inspect.getsource(advance_time_jax))


# In[11]:


def advance_time_jax(u, epsilon):
    """Advances the matrix by one timestep using JAX to do jit execution. It will be the same 
    process as the numpy version, but the compilation is jit.
    """
    j_padded_u = jnp.pad(u, 1, mode='constant', constant_values=0)
    right_roll = jnp.roll(j_padded_u, 1)
    left_roll = jnp.roll(j_padded_u, -1)
    top_roll = jnp.roll(j_padded_u, -1, axis=0)
    down_roll = jnp.roll(j_padded_u, 1, axis=0)
    j_updated_u = j_padded_u + epsilon*(right_roll + left_roll + top_roll + down_roll - (4*j_padded_u))
    j_updated_u_sliced = j_updated_u[1:-1, 1:-1]
    
    return j_updated_u_sliced


# We create an original array like before, and we update the matrix using computations in numpy itself.

# In[12]:


jitted_jax = jax.jit(advance_time_jax)
jnp_u = u0
# Initialize list to store results for visualization
results_jnp = []

# Measure start time
start_time = time.time()

# Run the simulation for 2700 iterations
for i in range(1,2701):
    jnp_u = jitted_jax(jnp_u, epsilon)
    if i % 300 == 0:
        results_jnp.append(jnp_u.copy())

# Calculate total time
execution_time = time.time() - start_time
print("Execution time:", execution_time, "seconds")

# Visualize the heat diffusion every 300 iterations
fig,axs = plt.subplots(3, 3, figsize=(12, 12))
count = 0
for i in range(3):
    for j in range(3):
        axs[i][j].imshow(results_jnp[count])
        axs[i][j].set_title(f"{(count+1)*300}th iteration")
        count += 1
    
plt.tight_layout()
plt.show()

