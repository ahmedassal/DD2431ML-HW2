#!/bin/sh

sudo apt-get update

sudo apt-get install libatlas-base-dev gfortran

sudo apt-get install python-dev

sudo apt-get install python-numpy python-scipy

sudo pip install cvxpy

sudo pip install nose

nosetests cvxpy

easy_install cvxopt

easy_install numpy

easy_install matplotlib
