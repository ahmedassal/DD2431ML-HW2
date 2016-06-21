from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy as np
import pylab as pl
import random as r
import math as m


def solve_for_alpha(P, q, G, h):
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    return list(r['x'])


def linearkernel(x1, x2):
    # print(np.dot(x1, x2))
    return np.dot(x1, x2) + 1


def polynomialkernel(x1, x2, power):
    # print(pow(np.dot(x1, x2)+1, power))

    if power > 0:
        #print(np.dot(x1, x2) + 1)
        return pow(np.dot(x1, x2) + 1, power) + 1


def radialkernel(x1, x2, sigma):
    # print(pow(np.dot(x1, x2)+1, power))
    #   print(2 * sigma**2)
    #print(np.asarray(x1))
    #print(np.asarray(x2))
    #print(np.linalg.norm(np.subtract(x1,x2)));
    diff = -1 * np.power( np.linalg.norm(np.subtract(x1,x2)), 2)
    #print(diff)
    return m.exp(-(np.dot(diff, diff) / (2 * sigma ** 2))) + 1


def sigmoidkernel(x1, x2, k, delta):
    # print(pow(np.dot(x1, x2)+1, power))
    #   print(k*np.dot(x1, x2) - delta)
    return m.tanh((k * np.dot(x1, x2) - delta) * m.pi / 180.0) + 1


def testlinearkernel():
    x1 = np.array([1, 2, 3])
    # print(x1)
    x2 = np.array([3, 2, 1])
    #   print(x2)
    return linearkernel(x1, x2)


def testpolynomialkernel(power):
    x1 = np.array([1, 2, 3])
    # print(x1)
    x2 = np.array([3, 2, 1])
    #   print(x2)
    return polynomialkernel(x1, x2, power)


def testradialkernel(sigma):
    x1 = np.array([2, 3, 1])
    # print(x1)
    x2 = np.array([3, 2, 1])
    #   print(x2)
    return radialkernel(x1, x2, sigma)


def testsigmoidkernel(k, delta):
    x1 = np.array([2, 3, 1])
    # print(x1)
    x2 = np.array([3, 2, 1])
    #   print(x2)
    return sigmoidkernel(x1, x2, k, delta)


def testkernels():
    print(testlinearkernel())
    print(testpolynomialkernel(2))
    print(testradialkernel(1))
    print(testsigmoidkernel(1, 0))



# linear kernel
def buildP(data, func, *args):

    datasize = len(data);
    #print(datasize);
    P = np.zeros((datasize, datasize));
    #print(type(P));
    #print(P);
    for i in range(0,datasize):
        for j in range(0,datasize):
            P[i][j] = data[i][2] * data[j][2] * func(data[i][0:2], data[j][0:2], *args);

    return P

def buildq(data):
    return -1 * np.ones(len(data));

def buildh(data):
    return np.zeros(len(data));

def buildG(data):
    G = np.zeros((len(data), len(data)));
    np.fill_diagonal(G, -1);
    return G;


def generateData(size):
    classA = [(r.normalvariate(-1.5, 1), r.normalvariate(0.5, 1), 1.0) for i in range(int(size / 2))] + \
             [(r.normalvariate(1.5, 1), r.normalvariate(0.5, 1), 1.0) for i in range(int(size / 2))];
    classB = [(r.normalvariate(0.0, 0.5), r.normalvariate(-0.5, 0.5), -1.0) for i in range(size)];
    data = classA + classB;
    r.shuffle(data);
    return data

def generateData2(size):
    classA = [(r.normalvariate(-1.0, .5), r.normalvariate(0.5, 1), 1.0) for i in range(int(size / 2))] + \
             [(r.normalvariate(1.0, .8), r.normalvariate(0.5, 1), 1.0) for i in range(int(size / 2))];
    classB = [(r.normalvariate(0.0, 0.5), r.normalvariate(-0.5, 0.5), -1.0) for i in range(size)];
    data = classA + classB;
    r.shuffle(data);
    return data

def generateData3(size):
    classA = [(r.normalvariate(-1.0, .5), r.normalvariate(0.5, 1), 1.0) for i in range(int(size))]

    classB = [(r.normalvariate(0.0, 0.5), r.normalvariate(-0.5, 0.5), -1.0) for i in range(size)];
    data = classA + classB;
    r.shuffle(data);
    return data

def plotData(data):
    pl.hold(True);
    pl.plot([d[0] for d in data if d[2] == 1], [d[1] for d in data if d[2] == 1], 'bo');
    pl.plot([d[0] for d in data if d[2] == -1], [d[1] for d in data if d[2] == -1], 'ro');
    #pl.show();
    pass

def indicator(testpoint, data, alphas, func, *args):
    sum = 0;
    for i in range(0, len(data)):
        sum = sum + alphas[i] * data[i][2] * func(testpoint, data[i][0:2], *args);
    return 1 if sum >=0 else -1;

def plotDecisionB(data, alphas, kernel, func, *args):
    xrange = np.arange(-4, 4, 0.05);
    yrange = np.arange(-4, 4, 0.05);

    grid = matrix([[indicator((x,y), data, alphas, func, *args) for y in yrange] for x in xrange]);
    #print(grid);

    pl.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1,3,1));
    pl.xlabel('x1');
    pl.ylabel('x2');
    if(kernel == 'linear'):
        pl.title('Linear Kernel');
    elif(kernel == 'polynomial'):
        pl.title('Polynomial Kernel');
    elif(kernel == 'radial'):
        pl.title('Radial Kernel');
    elif(kernel == 'sigmoidal'):
        pl.title('Sigmoidal Kernel');

    pl.show();

def run1(dataSize, dataSetNum, kernel):
    print('run 1');

    if (dataSetNum ==1):
        data = generateData(dataSize);
    elif(dataSetNum ==2):
        data = generateData2(dataSize);

    if(kernel == 'linear'):
        P = buildP(data, linearkernel);
    elif(kernel == 'polynomial'):
        P = buildP(data, polynomialkernel, 3);
    elif(kernel == 'radial'):
        P = buildP(data, radialkernel, 4);
    elif(kernel == 'sigmoidal'):
        P = buildP(data, sigmoidkernel, 3, 2);

    # print(matrix(P).size);

    q = buildq(data);
    # print(matrix(q).size);
    h = buildh(data);
    # print(matrix(h).size);
    G = buildG(data);
    # print(matrix(G).size);

    alphas = solve_for_alpha(P, q, G, h);
    alphas = [x for x in alphas if x > 10 ^-5]
    # print("alphas > 0 :" + alphas.__str__());
    # print(len(alphas))

    #print(len(data[0]))
    #testpoints = np.zeros((5, len(data[0])));
    #testpoints[0] = [5,5,1];
    #testtarget = indicator(testpoints[0], data, alphas, radialkernel, 2 );
    #print(testtarget);

    plotData(data);

    if(kernel == 'linear'):
        plotDecisionB(data, alphas, kernel, linearkernel);
    elif(kernel == 'polynomial'):
        plotDecisionB(data, alphas, kernel, polynomialkernel, 3);
    elif(kernel == 'radial'):
        plotDecisionB(data, alphas, kernel, radialkernel, 2 );
    elif(kernel == 'sigmoidal'):
        plotDecisionB(data, alphas, kernel, sigmoidkernel, 2, 3 );
    pass

def run2(dataSize, dataSetNum, kernel):

    print('run 2');
    if (dataSetNum ==1):
        data = generateData(dataSize);
    elif(dataSetNum ==2):
        data = generateData2(dataSize);

    if(kernel == 'linear'):
        P = buildP(data, linearkernel);
    elif(kernel == 'polynomial'):
        P = buildP(data, polynomialkernel, 7);
    elif(kernel == 'radial'):
        P = buildP(data, radialkernel, 8);
    elif(kernel == 'sigmoidal'):
        P = buildP(data, sigmoidkernel, 1, 6);

    # print(matrix(P).size);

    q = buildq(data);
    # print(matrix(q).size);
    h = buildh(data);
    # print(matrix(h).size);
    G = buildG(data);
    # print(matrix(G).size);

    alphas = solve_for_alpha(P, q, G, h);
    alphas = [x for x in alphas if x > 10 ^-5]
    # print("alphas > 0 :" + alphas.__str__());
    # print(len(alphas))

    #print(len(data[0]))
    #testpoints = np.zeros((5, len(data[0])));
    #testpoints[0] = [5,5,1];
    #testtarget = indicator(testpoints[0], data, alphas, radialkernel, 2 );
    #print(testtarget);

    plotData(data);

    if(kernel == 'linear'):
        plotDecisionB(data, alphas, kernel, linearkernel);
    elif(kernel == 'polynomial'):
        plotDecisionB(data, alphas, kernel, polynomialkernel, 3);
    elif(kernel == 'radial'):
        plotDecisionB(data, alphas, kernel, radialkernel, 2 );
    elif(kernel == 'sigmoidal'):
        plotDecisionB(data, alphas, kernel, sigmoidkernel, 3, 2 );
    pass

def run3(dataSize, dataSetNum, kernel):

    print('run 3');
    if (dataSetNum ==1):
        data = generateData(dataSize);
    elif(dataSetNum ==2):
        data = generateData2(dataSize);
    elif(dataSetNum ==3):
        data = generateData3(dataSize);

    if(kernel == 'linear'):
        P = buildP(data, linearkernel);
    elif(kernel == 'polynomial'):
        P = buildP(data, polynomialkernel, 3);
    elif(kernel == 'radial'):
        P = buildP(data, radialkernel, 2);
    elif(kernel == 'sigmoidal'):
        P = buildP(data, sigmoidkernel, 1, 6);

    # print(matrix(P).size);

    q = buildq(data);
    # print(matrix(q).size);
    h = buildh(data);
    # print(matrix(h).size);
    G = buildG(data);
    # print(matrix(G).size);

    alphas = solve_for_alpha(P, q, G, h);
    alphas = [x for x in alphas if x > 10 ^-5]
    # print("alphas > 0 :" + alphas.__str__());
    # print(len(alphas))

    #print(len(data[0]))
    #testpoints = np.zeros((5, len(data[0])));
    #testpoints[0] = [5,5,1];
    #testtarget = indicator(testpoints[0], data, alphas, radialkernel, 2 );
    #print(testtarget);

    plotData(data);

    if(kernel == 'linear'):
        plotDecisionB(data, alphas, kernel, linearkernel);
    elif(kernel == 'polynomial'):
        plotDecisionB(data, alphas, kernel, polynomialkernel, 3);
    elif(kernel == 'radial'):
        plotDecisionB(data, alphas, kernel, radialkernel, 2 );
    elif(kernel == 'sigmoidal'):
        plotDecisionB(data, alphas, kernel, sigmoidkernel, 3, 2 );
    pass
# main

#testkernels();
#run1(dataSize=10,dataSetNum = 1, kernel = 'radial');
run2(dataSize=10,dataSetNum = 2, kernel = 'polynomial');
#run3(dataSize=10,dataSetNum = 3, kernel = 'linear');

#kernel = 'polynomial';
#kernel = 'radial';
#kernel = 'sigmoidal';



















