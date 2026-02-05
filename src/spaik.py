''' 

Main program for

SPaiK   - Scalable Pairwise Kernel Learning Software using stochastic 
limited memory bundle method (StoLMBM), stochastic generalized vec 
trick (sGVT), and kernels from RLScore.  
                                                                       
The work was financially supported by the Research Council of Finland 
(Project No. #340140, #340182, #345804 and #345805).

The SPaiK software is covered by the MIT license.


First, select the data and loss function below. Then, run Makefile (by typing 
"make") to build a shared library that allows SLMBA (Fortran95 code) to be 
called from Python program SPaiK. The source uses f2py, Python3.7, and 
requires a Fortran  compiler (gfortran by default) and the RLScore 
 
https://github.com/aatapa/RLScore

to be installed. Finally, just type "python3.7 spaik.py". 


References:

    for SPaiK, sGVT, and StoLMBM:
       N. Karmitsa, T. Pahikkala, A. Airola "Scalable pairwise kernel learning 
       with stochastic vec trick", 2025. 

    for RLScore:
       T. Pahikkala, A. Airola, "Rlscore: Regularized least-squares learners", 
       Journal of Machine Learning Research, Vol. 17, No. 221, pp. 1-5, 2016.

    for LMBM and SLMBA :
       N. Haarala, K. Miettinen, M.M. Mäkelä, "Globally Convergent Limited Memory Bundle Method  
       for Large-Scale Nonsmooth Optimization", Mathematical Programming, Vol. 109, No. 1,
       pp. 181-205, 2007. DOI 10.1007/s10107-006-0728-2.

       M. Haarala, K. Miettinen, M.M. Mäkelä, "New Limited Memory Bundle Method for Large-Scale 
       Nonsmooth Optimization", Optimization Methods and Software, Vol. 19, No. 6, pp. 673-692, 2004. 
       DOI 10.1080/10556780410001689225.

       N. Karmitsa, V.-P. Eronen, M.M. Mäkelä, T. Pahikkala, A. Airola 
       "Stochastic limited memory bundle algorithm for clustering in big data", 
       Pattern Recognition, Vol 165, 111654, 2025. (A different version of the stochastic LMBM).


    for Nonsmooth Optimization:
       A. Bagirov, N. Karmitsa, M.M. Mäkelä, "Introduction to nonsmooth optimization: theory, 
       practice and software", Springer, 2014.


    for Interaction Concordance Index, Data, and Experimental Settings:
        T. Pahikkala, R. Numminen, P. Movahedi, N. Karmitsa, and A. Airola, "Interaction Concordance Index:
        Performance Evaluation for Interaction Prediction Methods", ArXiv2510.14419, 2025.


'''
import csv
import itertools as it
import multiprocessing as mp
import numpy as np
import time 

from numpy.random import SeedSequence
from rlscore.kernel import GaussianKernel, LinearKernel
from rlscore.measure import cindex,sqerror
from A_index import cython_assignmentIndex # Interaction concordance index

import data         # load pairwise data and divide it to different settings
import spaik        # fortran program
import pkl_utility  # python utility programs for SPaiK

def run_spkl(params):
    Y = params[0]
    XD = params[1]
    XT = params[2]
    drug_inds = params[3]
    target_inds = params[4]
    training_inds = params[5][0]
    test_inds = params[5][1]
    validation_inds = params[5][2]
    loss = params[6]
    kernels = params[7]
    ireg = params[8]
    autoreg = params[9]
    regparams = params[10]
    spaik.initpkl.epsilon = params[11]
    spaik.initpkl.ibin = params[12]
    bsize = params[13]
    seedOrig = params[14]
    

    """Reprocessing data"""

    # Indices for training, validation, and test.    
    train_drug_inds = drug_inds[training_inds]
    train_target_inds = target_inds[training_inds]
    Y_train = Y[training_inds]

    test_drug_inds = drug_inds[test_inds]
    test_target_inds = target_inds[test_inds]
    Y_test = Y[test_inds]
    
    validation_drug_inds = drug_inds[validation_inds]
    validation_target_inds = target_inds[validation_inds]
    Y_validation = Y[validation_inds]

    # The used of training set experimental settings IDIT, IDOT, ODIT, and ODOT 
    # are not explicitly given so they need to be found out.
    if set(test_drug_inds).isdisjoint(set(train_drug_inds)):
        if set(test_target_inds).isdisjoint(set(train_target_inds)):
            setting = "ODOT"
        else:
            setting = "ODIT"
    else:
        if set(test_target_inds).isdisjoint(set(train_target_inds)):
            setting = "IDOT"
        else:
            setting = "IDIT"

    
    """ Defining kernels """

    # Compute kernel matrices for drugs and targets
    drug_kernel_type = kernels[0][0]
    if drug_kernel_type == "linear":
        drug_kernel = LinearKernel(XD)
    elif drug_kernel_type == "gaussian":
        drug_kernel = GaussianKernel(XD, gamma=10**-5) # gamma=10**-5 is default
    KD = drug_kernel.getKM(XD)

    target_kernel_type = kernels[0][1]
    if target_kernel_type == "linear":
        target_kernel = LinearKernel(XT)
    elif target_kernel_type == "gaussian":
        target_kernel = GaussianKernel(XT, gamma=10**-5) # gamma=10**-5 is default
    KT = target_kernel.getKM(XT)

    # Create training, validation and test kernels (separate).  No retraining after validation.
    KD_train, rows1, rows2 = pkl_utility.K_to_dense(KD, train_drug_inds, train_drug_inds)
    KT_train, cols1, cols2 = pkl_utility.K_to_dense(KT, train_target_inds, train_target_inds)
    KD_val, rows_val1, rows_val2 = pkl_utility.K_to_dense(KD, validation_drug_inds, train_drug_inds)
    KT_val, cols_val1, cols_val2 = pkl_utility.K_to_dense(KT, validation_target_inds, train_target_inds)
    KD_test, rows_test1, rows_test2 = pkl_utility.K_to_dense(KD, test_drug_inds, train_drug_inds)
    KT_test, cols_test1, cols_test2 = pkl_utility.K_to_dense(KT, test_target_inds, train_target_inds)

    # Pairwise kernels
    pko_function = kernels[0][2]
    #pkotrain = eval('pkl_utility.'+pko_function+'(KD_train, KT_train, rows1, cols1, rows2, cols2)')
    pkoval =  eval('pkl_utility.'+pko_function+'(KD_val, KT_val, rows_val1, cols_val1, rows_val2, cols_val2)')     
    pkotest = eval('pkl_utility.'+pko_function+'(KD_test, KT_test, rows_test1, cols_test1, rows_test2, cols_test2)')
    

    """ Data and parameters """

    # Labels and drug and target kernels in fortran compatible form
    Y = np.array(Y_train, dtype='d', order='F')
    matD = np.array(KD_train, dtype='d', order='F')
    matT = np.array(KT_train, dtype='d', order='F')

    nrec = len(Y) 
    dw = 0 # Order in which the batch is selected. dw = 0 -> target-wise order, dw = 1 -> drugwise order.
    if dw == 1: # drug-wise order
        spaik.initpkl.m = len(KD_train)
        spaik.initpkl.q = len(KT_train)
        print (" The number of variables in optimization: ",nrec)
        print (" The numbers of unigue drugs and targets: ",spaik.initpkl.m, spaik.initpkl.q)
        print ('\n')

    else: # target-wise order
        spaik.initpkl.q = len(KD_train) # Note! Here q is number of drugs and m is number of targets
        spaik.initpkl.m = len(KT_train)
        print (" The number of variables in optimization: ",nrec)
        print (" The numbers of unigue drugs and targets: ",spaik.initpkl.q, spaik.initpkl.m)
        print ('\n')

    # Auxiliary matrices for sGVT
    size = spaik.initpkl.m * spaik.initpkl.q
    MM = np.zeros(size).reshape(spaik.initpkl.q,spaik.initpkl.m)
    matM = np.array(MM, dtype='d', order='F')
    matG = np.array(MM, dtype='d', order='F')

    batchsize = max(2, (bsize * spaik.initpkl.m) // 100 ) # Floor division
    iterm = np.array(0,dtype=np.int32) # This needs to be an array in order to change it in fortran
    nit = np.array(0,dtype=np.int32) # This needs to be an array in order to change it in fortran

    # regularization parameters
    spaik.initpkl.ireg = ireg # type of regularization
    if autoreg == 1: #  Use automatic determination of regularization parameter.
        spaik.initpkl.autolambda = 1
        nrho = 1 #
        spaik.initpkl.rho = 0.0
        spaik.initpkl.rho2 = 0.0
    else: #  Use a given list of regularization parameters.
        spaik.initpkl.autolambda = 0
        nrho = len(regparams)
    

    """ Initialization of indices """

    CIbest_of_best_vali = 0.0
    usedTimeAll = 0.0
    seed = [seedOrig]*8 # Seed for StoLMBM and sGVT
    
    for rp in range(nrho):
        if autoreg == 0:
            #print("Applying lambda ",regparams[rp],":\n")
            spaik.initpkl.rho = np.float32(regparams[rp]) 
            spaik.initpkl.rho2 = spaik.initpkl.rho
            
        # Starting time
        usedtime0 = time.process_time()

        # Starting point
        apy = np.zeros(int(nrec), dtype='d', order='F') # initialization of dual variable for Fortran
        ppy = np.zeros(int(nrec), dtype='d', order='F') # initialization of dual variable for Fortran
    
        # Compute C-index
        P_test = pkotest.matvec(apy)
        testCI = cindex(Y_test, P_test)
        
        # Compute MSE
        testMSE = sqerror(Y_test, P_test)

        # Initialization of indices
        CIbestvali = 0.0
        
        for h in range(10): # itmax^SPaiK = 10
            if dw == 1:
                spaik.fmodule.spaik(apy,ppy,Y,matD,matT,matM,matG,rows1,cols1,loss,iterm,nit,nrec,spaik.initpkl.m,spaik.initpkl.q,batchsize,seed,h)
            else:
                spaik.fmodule.spaik(apy,ppy,Y,matT,matD,matM,matG,cols1,rows1,loss,iterm,nit,nrec,spaik.initpkl.m,spaik.initpkl.q,batchsize,seed,h)

            P_val = pkoval.matvec(apy)
            P_test = pkotest.matvec(apy)

            # Compute C-index
            valiCI = cindex(Y_validation, P_val)
            #testCI = cindex(Y_test, P_test) 
            print("\n C-index in %s validation data with setting %s and lambda %.2e after %i iterations:  %f" %(ds, setting, spaik.initpkl.rho, nit, valiCI))
            #print("C-index in %s test data  with setting %s and lambda %2e after %i iterations:        %f" %(ds, setting, spaik.initpkl.rho, nit, testCI))
        
            ## Compute MSE
            ## If you want to compute MSE at every iteration uncomment the following lines. 
            ## Otherwise, MSE is computed in best-so-far solutions.
            #valiMSE = sqerror(Y_validation, P_val)
            #testMSE = sqerror(Y_test, P_test)
            #print("MSE in validation data with setting %s after %i iterations: %f" %(loss, setting, nit, valiMSE))
            #print("MSE in test data with setting %s after %i iterations:       %f" %(loss, setting, nit, testMSE))

            ## Compute IC-index
            ## If you want to compute IC-index at every iteration uncomment the following lines. 
            ## Otherwise, IC-index is computed in best-so-far solutions.
            #valiAI = cython_assignmentIndex(validation_drug_inds, validation_target_inds, Y_validation, P_val.reshape((P_val.shape[0],1)))[0]
            #testAI = cython_assignmentIndex(test_drug_inds, test_target_inds, Y_test, P_test.reshape((P_test.shape[0],1)))[0]
            #print("IC-index in validation data with setting %s after %i iterations: %f" %(loss, setting, nit, valiAI))
            #print("IC-index in test data with setting %s after %i iterations:       %f" %(loss, setting, nit, testAI))
            
            ## Additional performance indices
            ## If you want to compute group performance C-indices w.r.t. drugs and targets uncomment the following lines
            #gp1 = pkl_utility.group_performance(cindex, Y_test, P_test, test_drug_inds)
            #print("C-index w.r.t. drugs: ",gp1)
            #gp2 = pkl_utility.group_performance(cindex, Y_test, P_test, test_target_inds)
            #print("C-index w.r.t. targets",gp2)


            usedtime = time.process_time() - usedtime0
            if CIbestvali < valiCI:
                CIbestvali = valiCI
                if CIbest_of_best_vali < CIbestvali:
                    CIbest_of_best_vali = valiCI
                    
                    ## Compute MSE
                    #testMSE = sqerror(Y_test, P_test)
                    ## Compute IC-index
                    #testAI = cython_assignmentIndex(test_drug_inds, test_target_inds, Y_test, P_test.reshape((P_test.shape[0],1)))[0]
                    
                    #CIbesttest = testCI
                    #MSEbestwithCI = testMSE
                    #AIbestwithCI = testAI
                    itbestCI = nit+0 
                    bestlamCI = np.float32(spaik.initpkl.rho)
                    PCI = P_test

            if ((iterm == 1 or iterm == 2 or iterm == 3 or iterm == 8) and h>0):
                print (" Optimization successfully terminated!")
                ## Compute MSE
                #testMSE = sqerror(Y_test, PCI)
                ## Compute IC-index
                #testAI = cython_assignmentIndex(test_drug_inds, test_target_inds, Y_test, PCI.reshape((PCI.shape[0],1)))[0]
                ## Compute CI    
                #testCI = cindex(Y_test, PCI) 
                    
                #print ("The final solution (C-index, IC-index, MSE) in %s, under the setting %s: %f, %f, %f"%(ds, setting, testCI, testAI, testMSE))
                print (' CPU time = %4.2f' %(usedTimeAll+usedtime))
                #print ('\n')
                break 
            # Early stopping
            # Uncomment, if you want to have early stopping when the result is not improving from the previous batch
            #if CIbestvali > valiCI: 
                #print (" Early stopping.")
                ## Compute MSE
                #testMSE = sqerror(Y_test, PCI)
                ## Compute IC-index
                #testAI = cython_assignmentIndex(test_drug_inds, test_target_inds, Y_test, PCI.reshape((PCI.shape[0],1)))[0]
                ## Compute CI    
                #testCI = cindex(Y_test, PCI) 
                #print ("The final solution (C-index, IC-index, MSE) in %s, under the setting %s: %f, %f, %f"%(ds, setting, testCI, testAI, testMSE))
                #print (' CPU time = %4.2f' %(usedTimeAll+usedtime)) 
                #print ('\n')
                #break 

            # Print the intermediate result
            print("\n Current solution (C-index in validation) in %s under setting %s : %f" %(ds, setting, valiCI))
            print(' CPU time = %4.2f' %(usedTimeAll+usedtime))
            print('\n')
            #print('\n')
            seed = [5+h]*8 # Just to change this between validation   

        usedTimeAll += usedtime
        #print('\n')

    usedtime = usedTimeAll

    # Save predictions
    field = ["Setting","Y_test", "P_test (CI)"]
    with open('SPaiK_predictions_'+str(bsize)+'_'+loss+'_'+ds+'_'+setting+'_'+str(random_seed)+'.csv', 'w') as f:
        writer = csv.writer(f, delimiter=";", lineterminator="\n")
        writer.writerow(field)
        for i in range(len(Y_test)):
            predi = [setting,Y_test[i],PCI[i]]
            writer.writerow(predi)

    # Compute MSE
    testMSE = sqerror(Y_test, PCI)
    # Compute IC-index
    testAI = cython_assignmentIndex(test_drug_inds, test_target_inds, Y_test, PCI.reshape((PCI.shape[0],1)))[0]
    # Compute CI    
    testCI = cindex(Y_test, PCI) 
    print ("\n The final solution in %s test data (C-index, IC-index, and MSE) under the setting %s: %f, %f, %f"%(ds, setting, testCI, testAI, testMSE))
    print('\n')


    return(setting,bestlamCI,testCI,testAI,testMSE,itbestCI,nit+0,usedtime) 

if __name__ == "__main__":
    base_seed = 12345
    repetitions = 5 # The number of different data splits
    ss = SeedSequence(base_seed)
    random_seeds = ss.generate_state(repetitions)
    #seeds = [3] # Seeds for batches, use only one list item if batchsize = 100 
    seeds = [3,12,123,1234,12345]
    
    # Select the data from the list below (or add your own with appropriate loading procedure)
    datasets = ["davis"]
    #datasets = ["metz"]
    #datasets = ["kiba"]
    #datasets = ["merget"]
    #datasets = ["GPCR"]
    #datasets = ["IC"]
    #datasets = ["E"]
    #datasets = ["davis","metz","kiba","merget","GPCR","IC","E"]
    ibin = 0 # ibin = 1 with binary data (for GPRC, IC, and E, set later on)

    # Select percentage of samples in training data with setting IDIT
    split_percentage = 1.0/3
    
    # Select the loss function from the list below (only one at the time!)
    # Only Epsilon-Insensitive Squared Loss and Squared loss has been tested.
    #loss = "RLS"                 # Squared loss 
    #loss = "L1"                  # Absolute loss
    #loss = "hinge-loss"          # Epsilon-Intensive Absolute Loss
    loss = "semi-squared-hinge"   # Epsilon-Intensive Squared Loss
    #loss = "squared-hinge"       # Squared Epsilon-Intensive Loss
    #loss = "svm-hinge"           # Use without auto-regularization with regparam = 0.00001
    #loss = "squared-svm"         # Use without auto-regularization with regparam = 0.00001

    # Select epsilon for epsilon intensive hinge-losses
    epsilon = 0.0001 

    # Select the kernels (KD, KT, K_pairwise) from the list below (only one combination at the time).
    kernels = [["gaussian", "gaussian", "pko_kronecker"]]
    #kernels = [["linear", "linear", "pko_linear"]]     # results are not convincing
    #kernels = [["linear", "linear", "pko_kronecker"]]  # results are not convincing


    # Select regularization
    ireg = 1 # Switch for regularization: 0 = double regularization with L1- and L2-norms, 1 = L1-norm (default), 2 = L2-norm. 
    autoreg = 1 # Type of regularization: 0 = selected regularization parameters (give values below), 1 = automatic regularization (default).
    regparam = [2.0**(-10)] # Used with autoreg == 0. Note, give at least one regparam even if you use autoreg = 1
    #regparam = [2.0**(-25),2.0**(-20),2.0**(-15),2.0**(-10), 2.0**(-5),2.0**(-2)] # Short list is usually enough.
    #regparam = [2.0**(-10), 2.0**(-5), 2.0**(-4), 2.0**(-3), 2.0**(-2), 2.0**(-1), 2.0**(0), 2.0**(1), \
    #    2.0**(2), 2.0**(3), 2.0**(4), 2.0**(5), 2.0**(10)]
        
    # Select the size of the batch (percentages of n: 100 is full batch, while 20 means that nbatch ~ n/5)
    # Note that the batch is selected target-wise, and thus, the final size of the batch may vary a little bit.
    batchsize = 20 # Note change "seeds" above if batchsize = 100

    for ds in datasets:
        
        # Load data
        XD, XT, Y, drug_inds, target_inds = eval('data.load_'+ds+'()')    
        n_D = XD.shape[0]
        n_T = XT.shape[0]

        print('\n')
        print('SPaiK:     Loss function:          ',loss)
        print('           Used data:              ',ds, 'with',n_D,'drugs and',n_T,'targets.')
        print('           Kernel for drugs:       ',kernels[0][0])
        print('           Kernel for targets:     ',kernels[0][1])
        print('           Pairwise kernel:        ',kernels[0][2])
        if autoreg == 1:
            if ireg == 0:
                print('           Regularization:         ',"double regularization with an automatic regularization parameter")
            elif ireg == 1:
                print('           Regularization:         ',"L1-norm with an automatic regularization parameter") 
            else:
                print('           Regularization:         ',"L2-norm with an automatic regularization parameter") 
        else:        
            if ireg == 0:
                print('           Regularization:         ',"double regularization with selected regularization parameters")
            elif ireg == 1:
                print('           Regularization:         ',"L1-norm with selected regularization parameters") 
            else:
                print('           Regularization:         ',"L2-norm with selected regularization parameters") 
        print('           Size of the batch:      ',batchsize,' percentages of all training labels.')

        print('\n')

        # Mark binary data as binary
        if ds == "GPRC" or ds == "IC" or ds == "E":
            ibin = 1
 
        # Output (give name of the output file with performance indices below)
        field = ["Data: "+ds+"."]
        outfile = 'test_'+str(batchsize)+'.csv'
        with open(outfile, 'a') as f:
            writer = csv.writer(f, delimiter=";", lineterminator="\n")
            writer.writerow(field)

        for random_seed in random_seeds: # Runs with different random splits of data
            field = ["Setting","lambda (CI)", "C-index (CI)","IC-index (CI)",  "MSE (CI)","nit (CI)","nit total","CPU-time"]
            with open(outfile, 'a') as f:
                writer = csv.writer(f, delimiter=";", lineterminator="\n")
                writer.writerow(field)
        
            time_start = time.time()
        
        # Create the splits for the four different experimental settings such that there is one common test set 
        # for every setting. The split_percentage defines number of training samples in S1 setting. The rest of 
        # samples are divided to test and validation.
            df, splits = data.splits(drug_inds, target_inds, split_percentage, random_seed)
            print("Splits in "+ds+" calculated in %.2f seconds." %(time.time()-time_start),'\n')
            splits_1234 = list(it.chain.from_iterable(splits))
                
            for seed in seeds: # Runs with different random batches
                parameters = it.product([Y], [XD], [XT], [drug_inds], [target_inds], splits_1234,[loss],[kernels],[ireg],[autoreg],[regparam],[epsilon],[ibin],[batchsize],[seed])
        
                # Compute different settings at the same time.
                pool = mp.Pool(processes = 4) # Note that the prints for console may be mixed
                #pool = mp.Pool(processes = 1)
                output = pool.map(run_spkl, list(parameters))
                pool.close()
                pool.join()

                print(output)
                print('\n')
                
                # Save result indices (predictions are saved above)
                with open(outfile, 'a') as f:
                    writer = csv.writer(f, delimiter=";", lineterminator="\n")
                    writer.writerows(output)
                

