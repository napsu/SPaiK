# SPaiK - Scalable Pairwise Kernel Learning Software

SPaiK is a scalable software package for pairwise kernel learning. It combines the stochastic limited-memory bundle method (StoLMBM) for optimization, the stochastic generalized vec trick (sGVT) for efficient computation with pairwise Kronecker kernels, and a rich set of kernel functions provided by RLScore.

The included loss-functions for the pairwise kernel problem are:
* squared loss,
* squared epsilon-insensitive loss,
* epsilon-insensitive squared loss,
* epsilon-insensitive absolute loss,
* absolute loss.

Note that only the epsilon-insensitive squared loss has been tested for functionality.

## Files included
    
* spaik.py              
  - Main Python file. Includes [RLScore](https://github.com/aatapa/RLScore) calls.
* pkl_utility.py        
  - Python utility programs.
* spaik.f95             
  - Building block between Python and Fortran for pairwise learning software.
* slmba.f95             
  - StoLMBM - a stochastic limited memory bundle method for nonsmooth optimization (specially developed for SPaiK).
* objfun.f95            
  - Computation of the function and subgradients values with different loss functions. Selection between loss functions is made in spaik.py. Includes sGVT.
* initpkl.f95           
  - Initialization of parameters and variables for SPaiK and StoLMBM. Includes modules:
    + initpkl - Initialization of parameters for SPaiK.
    + initslmba - Initialization of StoLMBM.
* parameters.f95
  - Parameters for Fortran. Inludes modules:
    + r_precision - Precision for reals,
    + param - Parameters.
* subpro.f95            
  - Subprograms for StoLMBM.
* data.py
  - Contains functions to load the example datasets in SPaiK. The data files are assumed to be located in a folder "data". This repository does not include the datasets themselves; links to all example datasets are provided
in repository [github.com/TurkuML](https://github.com/TurkuML/IC-index-experiments).

  - Contains functions to create train-test-validation splits. Splits are created for every experimental setting IDIT, IDOT, ODIT, and ODOT (see the references below).

* Makefile              
  - makefile: Builds a shared library to allow StoLMBM (Fortran95 code) to be called from Python program SPaiK. Uses f2py, Python3.7, and requires a Fortran compiler (gfortran) to be installed.



## Installation and usage
The source uses f2py and Python3.7, and requires a Fortran  compiler (gfortran by default) and the [RLScore](https://github.com/aatapa/RLScore) to be installed.

To use the code:
1) Select the data and loss function from spaik.py file.
2) Run Makefile (by typing "make") to build a shared library that allows spaik.f95 (Fortran95 code) to be called from Python program spaik.py. 
3) Finally, just type "python3.7 spaik.py".


The algorithm returns a csv-file with performance measures (C-index, IC-index, and MSE) computed in the test set under different experimental settings IDIT, IDOT, ODIT, and ODOT. The best results are selected using a separate validation set and validated w.r.t. C-index.
In addition, separate csv-files with predictions under different experimental settings are returned. 

## References:

* SPaiK, sGVT, and StoLMBM:
  - N. Karmitsa, T. Pahikkala, A. Airola, "Scalable Pairwise Kernel Learning 
       with Stochastic Vec Trick", 2025. 
* [RLScore](https://github.com/aatapa/RLScore):
  - T. Pahikkala, A. Airola, "[Rlscore: Regularized Least-Squares Learners](https://www.jmlr.org/papers/volume17/16-470/16-470.pdf)", Journal of Machine Learning Research, Vol. 17, No. 221, pp. 1-5, 2016.
* LMBM and SLMBA:
  - N. Haarala, K. Miettinen, M.M. Mäkelä, "[Globally Convergent Limited Memory Bundle Method for Large-Scale Nonsmooth Optimization](https://link.springer.com/article/10.1007/s10107-006-0728-2)", Mathematical Programming, Vol. 109, No. 1, pp. 181-205, 2007.
  - M. Haarala, K. Miettinen, M.M. Mäkelä, "[New Limited Memory Bundle Method for Large-Scale Nonsmooth Optimization](https://www.tandfonline.com/doi/abs/10.1080/10556780410001689225)", Optimization Methods and Software, Vol. 19, No. 6, pp. 673-692, 2004.
  - N. Karmitsa, V.-P. Eronen, M.M. Mäkelä, T. Pahikkala, A. Airola, 
       "[Stochastic Limited Memory Bundle Algorithm for Clustering in Big Data](https://www.sciencedirect.com/science/article/pii/S0031320325003140?via%3Dihub)", 
       Pattern Recognition, Vol 165, 111654, 2025. (A different version of the Stochastic LMBM).
* Generalized vec trick:
  - A. Airola, T. Pahikkala, "[Fast Kronecker Product Kernel Methods via Generalized Vec Trick](https://ieeexplore.ieee.org/document/7999226)", IEEE Transactions on Neural Networks and Learning Systems, Vol. 29, pp. 3374–3387, 2018.
  - M. Viljanen, A. Airola, T. Pahikkala, "[Generalized Vec Trick for Fast Learning of Pairwise Kernel Models](https://link.springer.com/article/10.1007/s10994-021-06127-y)", Machine Learning, Vol. 111, 543–573, 2022.
* Nonsmooth optimization:
  - A. Bagirov, N. Karmitsa, M.M. Mäkelä, "[Introduction to Nonsmooth Optimization: Theory, Practice and Software](https://link.springer.com/book/10.1007/978-3-319-08114-4)", Springer , 2014.
* [Interaction-Concordance-Index](https://github.com/TurkuML/Interaction-Concordance-Index), data, and experimental settings:
  - T. Pahikkala, R. Numminen, P. Movahedi, N. Karmitsa, and A. Airola, "[Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods](https://arxiv.org/abs/2510.14419)", ArXiv2510.14419, 2025.


## Acknowledgements
The work was financially supported by the Research Council of Finland, Project No. #340182 and
#345804 led by Tapio Pahikkala and Project No. #340140 and #345805 led by Antti Airola.


   
