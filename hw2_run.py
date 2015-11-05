from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy as np
import pylab as pl
import random as r
import math as m


def solve_for_alpha(P, q, G, h):
  r =  qp(matrix(P), matrix(q), matrix(G), matrix(h))
  return list(r['x'])


def linearkernel(x1, x2):
#   print(np.dot(x1, x2))
  return np.dot(x1, x2) + 1

def polynomialkernel(x1, x2, power):
#   print(pow(np.dot(x1, x2)+1, power))
  if power > 0:
    return pow(np.dot(x1, x2)+1, power)

def radialkernel(x1, x2, sigma):
#   print(pow(np.dot(x1, x2)+1, power))
#   print(2 * sigma**2)
  diff = -pow(x1-x2+1, 2)
#   print(diff)
  return m.exp(-np.sqrt(diff.dot(diff))/(2 * sigma**2))

def sigmoidkernel(x1, x2, k, delta):
#   print(pow(np.dot(x1, x2)+1, power))
#   print(k*np.dot(x1, x2) - delta)
  return m.tanh((k*np.dot(x1, x2) - delta) * m.pi / 180.0)



def testlinearkernel():
  x1 = np.array([2,3,1])
#   print(x1)
  x2 = np.array([1,5,10])
#   print(x2)
  return linearkernel(x1, x2)

def testpolynomialkernel(power):
  x1 = np.array([2,3,1])
#   print(x1)
  x2 = np.array([1,5,10])
#   print(x2)
  return polynomialkernel(x1, x2, power)

def testradialkernel(sigma):
  x1 = np.array([2,3,1])
#   print(x1)
  x2 = np.array([1,5,10])
#   print(x2)
  return radialkernel(x1, x2, sigma)

def testsigmoidkernel(k, delta):
  x1 = np.array([2,3,1])
#   print(x1)
  x2 = np.array([1,5,10])
#   print(x2)
  return sigmoidkernel(x1, x2, k, delta)


def testkernels():
  print(testlinearkernel())
  print(testpolynomialkernel(3))
  print(testradialkernel(1))
  print(testsigmoidkernel(1,0))


testkernels()