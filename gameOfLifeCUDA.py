import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import cv2

# CUDA Kernel
kernel_code = """
__global__ void updateGameOfLife(int ny, int nx, unsigned char *d_grid, unsigned char *d_newGrid) 
{
     int row = blockIdx.x * blockDim.x + threadIdx.x;
     int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < ny && col < nx) 
    {
        int numNeighbors = 0;
        
        // Calculate the number of alive neighbors
        for (int y = -1; y <= 1; y++)
        {
            for (int x = -1; x <= 1; x++)
            {
                if (y == 0 && x == 0) continue; // Skip the cell itself
                int neighborCol = (col + x + nx) % nx;
                int neighborRow = (row + y + ny) % ny;
                numNeighbors += (d_grid[neighborRow * nx + neighborCol] == 1); // Count as neighbor if cell is alive (1)
                
            }
        }

        // Apply Game of Life rules
        // Rule 1: Any live cell with two or three live neighbors survives.
        // Rule 2: Any dead cell with three live neighbors becomes a live cell.
        // Rule 3: All other live cells die in the next generation. Similarly, all other dead cells stay dead.
        unsigned char cell = d_grid[row * nx + col];
        unsigned char newCell = (cell == 1 && (numNeighbors == 2 || numNeighbors == 3)) || (cell == 0 && numNeighbors == 3) ? 1 : 0;
        d_newGrid[row * nx + col] = newCell;
    }
}
"""

mod = SourceModule(kernel_code)
updateGameOfLife = mod.get_function("updateGameOfLife")

ny, nx, maxiter = 1024, 1024, 100  # example values
blockSize = (16, 16, 1)
gridSize = ((nx + blockSize[0] - 1) // blockSize[0], (ny + blockSize[1] - 1) // blockSize[1], 1)

# Initialize population
population = np.random.randint(2, size=(ny * nx), dtype=np.uint8)

# Allocate memory on GPU
d_grid = drv.mem_alloc(population.nbytes)
d_newGrid = drv.mem_alloc(population.nbytes)
drv.memcpy_htod(d_grid, population)

# Main loop
for iter in range(maxiter):
    updateGameOfLife(np.int32(ny), np.int32(nx), d_grid, d_newGrid, block=blockSize, grid=gridSize)

    # Copy data back to host
    drv.memcpy_dtoh(population, d_newGrid)
    
    # Display using OpenCV
    img = population.reshape((ny, nx)).astype(np.uint8) * 255
    cv2.imshow("Game of Life", img)
    cv2.waitKey(30)

    # Swap pointers
    d_grid, d_newGrid = d_newGrid, d_grid

cv2.destroyAllWindows()
