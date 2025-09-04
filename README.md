# SPaiK - Scalable Pairwise Kernel Learning Software

SPaiK is a scalable software package for pairwise kernel learning. It combines the stochastic limited-memory bundle algorithm (SLMBA) for optimization, the stochastic generalized vec trick (sGVT) for efficient computation with pairwise Kronecker kernels, and a rich set of kernel functions provided by RLScore.

The included loss-functions for the pairwise kernel problem are:
* squared loss,
* squared epsilon-insensitive loss,
* epsilon-insensitive squared loss,
* epsilon-insensitive absolute loss,
* absolute loss.



## Files included
    
* spaik.py              
  - Main Python file. Includes [RLScore](https://github.com/aatapa/RLScore) calls.
* pkl_utility.py        
  - Python utility programs.
* spaik.f95             
  - Building plock between Python and Fortran for pairwise learning software.
* slmba.f95             
  - SLMBA - a stochastic limited memory bundle algorithm for nonsmooth optimization (specially modified for SPaiK).
* objfun.f95            
  - computation of the function and subgradients values with different loss functions. Selection between loss functions is made in spaik.py. In addition, includes sGVT.
* initpkl.f95           
  - initialization of parameters and variables for SPaiK and SLMBA. Includes modules:
    + initpkl - Initialization of parameters for SPaiK.
    + initslmba - Initialization of SLMBA.
* parameters.f95
  - Parameters for Fortran. Inludes modules:
    + r_precision - Precision for reals,
    + param - Parameters.
* subpro.f95            
  - subprograms for SLMBA.
* data.py
  - Contains functions to load the example data sets. Data files are assumed to be in a folder "data" that is not part of the current folder.
  - Contains functions to create train-test-validation splits. Splits are created for every experimental setting S1-S4 (see the reference below).

* Makefile              
  - makefile: builds a shared library to allow SLMBA (Fortran95 code) to be called from Python program spaik. Uses f2py, Python3.7, and requires a Fortran compiler (gfortran) to be installed.



## Installation and usage
The source uses f2py and Python3.7, and requires a Fortran  compiler (gfortran by default) and the [RLScore](https://github.com/aatapa/RLScore) to be installed.

To use the code:
1) Select the data and loss function from spaik.py file.
2) Run Makefile (by typing "make") to build a shared library that allows SLMBA (Fortran95 code) to be called from Python program spaik. 
3) Finally, just type "python3.7 spaik.py".


The algorithm returns a csv-file with performance measures (C-index, IC-index, and MSE) computed in the test set under different experimental settings S1-S4. The best results are selected using a separate validation set and validated w.r.t. C-index.
In addition, separate csv-files with predictions under different experimental settings S1-S4 are returned. 

## References:

* SPaiK, sGVT, and SLMBA:
  - N. Karmitsa, T. Pahikkala, A. Airola "Scalable pairwise kernel learning 
       with stochastic vec trick", 2025. 
* [RLScore](https://github.com/aatapa/RLScore):
  - T. Pahikkala, A. Airola, "[Rlscore: Regularized least-squares learners](https://www.jmlr.org/papers/volume17/16-470/16-470.pdf)", Journal of Machine Learning Research, Vol. 17, No. 221, pp. 1-5, 2016.
* SLMA and LMBM:
  - N. Karmitsa, V.-P. Eronen, M.M. Mäkelä, T. Pahikkala, A. Airola 
       "[Stochastic limited memory bundle algorithm for clustering in big data](https://www.sciencedirect.com/science/article/pii/S0031320325003140?via%3Dihub)", 
       Pattern Recognition, Vol 165, 111654, 2025. (A slightly different version of the SLMBA).
  - N. Haarala, K. Miettinen, M.M. Mäkelä, "[Globally Convergent Limited Memory Bundle Method for Large-Scale Nonsmooth Optimization](https://link.springer.com/article/10.1007/s10107-006-0728-2)", Mathematical Programming, Vol. 109, No. 1, pp. 181-205, 2007.
  - M. Haarala, K. Miettinen, M.M. Mäkelä, "[New Limited Memory Bundle Method for Large-Scale Nonsmooth Optimization](https://www.tandfonline.com/doi/abs/10.1080/10556780410001689225)", Optimization Methods and Software, Vol. 19, No. 6, pp. 673-692, 2004.
* Generalized vec trick and experimental settings:
  - A. Airola, T. Pahikkala, "[Fast kronecker product kernel methods via generalized vec trick](https://ieeexplore.ieee.org/document/7999226)", IEEE Transactions on Neural Networks and Learning Systems, Vol. 29, pp. 3374–3387, 2018.
  - M. Viljanen, A. Airola, T. Pahikkala, "[Generalized vec trick for fast learning of pairwise kernel models](https://link.springer.com/article/10.1007/s10994-021-06127-y)", Machine Learning, Vol. 111, 543–573, 2022.
* Nonsmooth optimization:
  - A. Bagirov, N. Karmitsa, M.M. Mäkelä, "[Introduction to nonsmooth optimization: theory, practice and software](https://link.springer.com/book/10.1007/978-3-319-08114-4)", Springer, 2014.


## Acknowledgements
The work was financially supported by the Research Council of Finland projects (Project No. #345804 and #345805) led by Antti Airola and Tapio Pahikkala.


   
