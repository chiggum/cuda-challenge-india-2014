# CUDA CHALLENGE INDIA 2017

* Winning solution

### Input
* A square matrix with some values in each cell.

### Algorithm Description:
#### Saturate maps:
* repeat until no hills and dales are left or MAX_ITERATIONS reached (hill is a cell with maximum value among the surrounding 8 cells and dale is a cell with minimum value among the surrounding 8 cells)
* One thread launched for each cell.
* 8 surrounding cells are checked.
* In case of hill, mean is replaced and in case of dale median is replaced.

#### Binarizing maps:
* Threshold is calculated using reduction operation on map elements with plus as reduction op. (Threshold is the mean of the values in the matrix)
* Then, one thread is launched per cell and each cell is compared with threshold.
* less than threshold: write 0 else 1 in that cell.

#### Number of Connected component calculation:
* The Reference papers describes the parallel algorithms for connected component labelling. I used the algorithm in second paper.

#### Reference papers:
* For Concept: Parallel Graph Component Labelling with GPUs and CUDA. K.A. Hawick, A. Leist and D.P. Playne
* For optimization: Connected component labeling on a 2D grid using CUDA. Oleksandr Kalentev, Abha Rai, Stefan Kemnitz and Ralf Schneider

#### Testing environment:
* GPU: Tesla K40
* OS: Ubuntu 12.04 Kernel:3.2.0
* CUDA Toolkit: 6.0.37
* Driver: 331.62
* CPU RAM: 16 GB
* PCI Gen2/3
