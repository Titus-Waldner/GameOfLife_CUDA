# GameOfLife_CUDA
A small project to experiemnet with cuda kernals.

![image](https://github.com/user-attachments/assets/a630ef01-e41d-47e1-ad5a-21164de90a2d)

A simple example of accelerating the Game of Life with CUDA in Python. In this cellular automaton, a matrix of cells evolves in each iteration based on the states of neighboring cells. The Python implementation retains the same CUDA kernel as in the C++ version from Lab 5, embedded as a string and compiled using PyCUDA's SourceModule. PyCUDA replaces C++ CUDA APIs for memory allocation and data transfer. Visualization shifts from C++ OpenCV's cv::Mat to Python OpenCV, displaying the grid as a 2D NumPy array with cv2.imshow. The code's main loop and initialization are adapted to Python syntax, with command-line inputs hardcoded for simplicity.
