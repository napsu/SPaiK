#*************************************************************************
#*                                                                       *
#*     Makefile for SPaiK --- A scalable pairwise kernel                 *
#*                               learning software                       *
#*                                                                       *
#*************************************************************************
#
# Build a shared library to allow SLMBA (Fortran95 code) to be called
# from Python program SPaiK. Uses f2py, Python3.7, and requires 
# a Fortran compiler (here gfortran) to be installed.
#
# After running this makefile, run the program by typing 
#   python3.7 spaik.py
# 
#FF = gfortran -O2 -ffast-math
#EXT_SUFFIX := $(zsh python3-config --extension-suffix)
EXT_SUFFIX := $(shell python3-config --extension-suffix)
FF = gfortran -O2
PF = python3.7 -m numpy.f2py
FLAGS = 
#FLAGS = --f90flags='-O2'
#FLAGS = --fcompiler=gfortran --f90flags='-fimplicit-none -O2'

all:	spaik$(EXT_SUFFIX)
#
# ---- Compile Fortran files --------------------------- 
parameters.o: parameters.f95
	$(FF) -c parameters.f95

initpkl.o: r_precision.mod initpkl.f95
	$(FF) -c initpkl.f95

objfun.o: r_precision.mod objfun.f95
	$(FF) -c objfun.f95

subpro.o: r_precision.mod param.mod subpro.f95
	$(FF) -c subpro.f95

slmba.o: r_precision.mod param.mod initpkl.mod initslmba.mod exe_time.mod obj_fun.mod subpro.mod slmba.f95
	$(FF) -c slmba.f95

# ---- Create Signature ------------------
spaik.pyf: parameters.o initpkl.o objfun.o subpro.o slmba.o spaik.f95
	$(PF) --overwrite-signature parameters.f95 initpkl.f95 objfun.f95 subpro.f95 slmba.f95 spaik.f95 -m spaik -h spaik.pyf

# ---- Compile and Link --------------------------- 
spaik$(EXT_SUFFIX): parameters.o initpkl.o objfun.o subpro.o slmba.o spaik.f95 spaik.pyf
	$(PF) -c $(FLAGS) spaik.pyf parameters.f95 initpkl.f95 objfun.f95 subpro.f95 slmba.f95 spaik.f95

# ---- Clean -----------------------------
clean:	
	rm -rf *.o *.mod *.so *.pyf __pycache__
	echo Clean done	