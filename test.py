import numpy as np
import math
from numba import cuda
import time
'''@cuda.jit
def increment_a_2D_array(a,l):
    x, y = cuda.grid(2)
   

    # # Thread id in a 1D block
    # tx = cuda.threadIdx.x
    # # Block id in a 1D grid
    # bx = cuda.blockIdx.x
    # # Thread id in a 1D block
    # ty = cuda.threadIdx.y
    # # Block id in a 1D grid
    # by = cuda.blockIdx.y
    # # Block width, i.e. number of threads per block
    # bw = cuda.blockDim.x
    # bh = cuda.blockDim.y
    # # Compute flattened index inside the array
    # x = tx + bx * bw
    # y = ty + by * bh

    # print(tx,ty, bx, by, bw, bh)
    nx,ny = x,0
    i=0
    if x < l.shape[0] and y < l.shape[1]:        
        l[nx, ny] += 1
    if x < a.shape[0] and y < a.shape[1]:  
        a[nx,ny] =1
        



data = np.zeros((35,100))
gdata = cuda.to_device(data)
# Configure the blocks
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(data.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(data.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
l=np.array((35,100))
gl = cuda.to_device(l)

increment_a_2D_array[blockspergrid, threadsperblock](gl, data)
print(data)
print(data.shape)
print(gl)'''

'''
@cuda.jit
def my_kernel2(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation
# from __future__ import division
from numba import cuda
import numpy


# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation

# Host code   
data = numpy.ones(100)
threadsperblock = 32
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](data)
print(data)
print(data.shape)'''



@cuda.jit(debug=True)
def adjust_log(a, b, c):
    ix, iy = cuda.grid(2) # The first index is the fastest dimension
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2) #  threads per grid dimension
    
    n0, n1,_ = a.shape # The last index is the fastest dimension
    # Stride each dimension independently
    for i0 in range(iy, n0, threads_per_grid_y):
        for i1 in range(ix, n1, threads_per_grid_x):
            c[i0, i1, 0] = b[i0,i1,0] +i0
            c[i0, i1, 1] = b[i0,i1,1] +i1

threads_per_block_2d = (16, 16)  #  256 threads total
blocks_per_grid_2d = (64, 64)

moon = np.zeros((512,512,2))
moon1 = np.zeros((512,512,2))
# moon_gpu = cuda.to_device(moon)

moon_corr_gpu = cuda.device_array_like(moon)
now=time.time()
adjust_log[blocks_per_grid_2d, threads_per_block_2d](moon, moon1, moon_corr_gpu)
after = time.time()
moon_corr = moon_corr_gpu.copy_to_host()

print(moon_corr)
print(moon_corr.shape)
print(after-now)


def adjust_cpu(a, b, c):
    n0, n1,_ = a.shape # The last index is the fastest dimension
    # Stride each dimension independently
    for i0 in range(0, n0):
        for i1 in range(0, n1):
            c[i0, i1, 0] = b[i0,i1,0] +i0
            c[i0, i1, 1] = b[i0,i1,1] +i1
c=np.zeros((512,512,2))
now=time.time()
adjust_cpu(moon, moon1, c)
after = time.time()
print(c)
print(c.shape)
print(after-now)