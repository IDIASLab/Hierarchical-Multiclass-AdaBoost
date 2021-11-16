#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:06:03 2020

@author: qwt
"""

from numpy import *


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D, check_list, hie_flag):

    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  
    for i in range(n):  
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps  
        for j in range(-1, int(numSteps) + 1):  
            for inequal in ['lt', 'gt']:  
                threshVal = (rangeMin + float(j) * stepSize) 
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) 
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                if (hie_flag):
                    errArr[check_list == 1] = 1 
                    check_list = mat(zeros((m, 1)))
                    check_list[errArr == 1] = 1
                else:
                    check_list[predictedVals == labelMat] = 0
                weightedError = D.T * errArr 
                
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst, check_list