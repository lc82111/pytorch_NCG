# pytorch_NCG

An embarrassing simple implementation for the nonlinear conjugate gradient in PyTorch.

The main idea is combining the auto gradient of PyTorch and nonlinear conjugate gradient algorithm of Scipy (scipy.optimize.fmin_cg).

The implementation is weak, because of the frequent data transfer between GPU and CPU.

Please consider it as a toy.

Please let me known, if you find any implementation of GC in PyTorch. 

Thanks.


Ref:

[1] An Introduction to the Conjugate Gradient Method Without the Agonizing Pain
