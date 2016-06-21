#!/bin/sh

sudo apt-get update

sudo apt-get install pkg-config

sudo apt-get install libatlas-base-dev gfortran

sudo apt-get install python-dev

sudo apt-get install python-numpy python-scipy

sudo pip3 install cvxpy

sudo pip3 install nose

nosetests cvxpy

easy_install3 cvxopt

easy_install3 numpy

easy_install3 matplotlib
