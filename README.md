# pytorch_NCG

A embarrassing simple implementation for the nonlinear conjugate gradient in pytorch.

The main ideas is combining the auto gradient of pytorch and nonlinear conjugate gradient algrithom of Scipy (scipy.optimize.fmin_cg).

The implementation is definitely weak, because of the frequent data transfer between GPU and CPU.

Please considering it as a toy.

Please let me known, if you find any implementation of GC in pytorch. 

Thanks.
