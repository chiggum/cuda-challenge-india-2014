# CUDA CHALLENGE INDIA

### Team Flash:
* Dhruv Kohli
* Vedant Kohli

### Algorithm Description:
#### Saturate maps
* repeat until no hills and dales are left or MAX_ITERATIONS reached
* One thread launched for each cell.
* 8 surrounding cells are checked.
* In case of hill,  mean is replaced and in case of dale median is replaced.

#### Binarizing maps:
* Threshold is calculated using reduction operation on map elements with plus as reduction op.
* Then, one thread is launched per cell and each cell is compared with threshold.
* less than threshold: write 0 else 1.

#### Number of Connected component calculation:
* CCL Algorithm used.
* Special Note: The serial version is too slow and it is not easy to come up with an algorithm in parallel for connected component labelling. The Reference papers describes the parallel algorithms very aptly and clearly.

#### Reference papers:
* For Concept: Parallel Graph Component Labelling with GPUs and CUDA. K.A. Hawick, A. Leist and D.P. Playne
* For optimization: Connected component labeling on a 2D grid using CUDA. Oleksandr Kalentev, Abha Rai, Stefan Kemnitz and Ralf Schneider
