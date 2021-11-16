#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:07:52 2020

@author: qwt
"""

from decision_stump import *
import numpy as np


class HAdaBoostMH:


    weak_classifiers = ['stump']
    __max_iter = 40

    def __init__(self, class_num, max_iter, hier_wei):
        self.__max_iter = max_iter
        self.__class_num = class_num
        self.hier_weight = hier_wei
        self.check_list = None

    __weakClassArr = []

    def fit(self, X, y, hier_num_label):
        self.__weakClassArr = []
        m = shape(X)[0]
        # m is total number of train set       
        initial = mat(ones((m, self.__class_num)) / (m * self.__class_num))
        tran_matrix = mat(np.diag(np.array(self.hier_weight)))
        
        
        # weight with hierarchichy
        D = mat(ones((m, self.__class_num)) / (m * self.__class_num))
       
        aggClassEst = mat(zeros((m, self.__class_num)))
        
        for i in range(self.__max_iter):
            bestArgsList = []          
            errorList = []
            classEstList = []
            classEstMat = ones((m, self.__class_num))
            label_num = 0
            flag = False
            for ii in range(len(hier_num_label)):
                if (ii == 0):
                    self.check_list = mat(ones((m, 1)))
                    
                else:
                    flag = True
#                    
                for j in range(len(hier_num_label[ii])):
                    bestArgs, error, classEst,  self.check_list = buildStump(X,  y[:, label_num].flatten().tolist(),
                                                           D[:, label_num], self.check_list, flag)
                    bestArgsList.append(bestArgs)
                    errorList.append(error)
                    classEstList.append(classEst)
                    classEstMat[:, label_num] = classEst.flatten()
                    label_num = label_num + 1
                    
                    
          
            error = sum(errorList)
            print ("current error: %.6f " % (error))
            print (i)

            alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
            for ii in range(self.__class_num):
                bestArgsList[ii]['alpha'] = alpha

            self.__weakClassArr.append(bestArgsList)  


            expon = multiply(-1 * alpha * mat(y), classEstMat)

            D = multiply(D, exp(expon))

            D = D / D.sum()
            aggClassEst += alpha * classEst
            aggErrors = multiply(sign(aggClassEst) != mat(y).T, ones((m, 1)))
            errorRate = aggErrors.sum() / m
            if errorRate == 0.0: break    
        return self.__weakClassArr, aggClassEst

    def predict(self, X, label_id):
        dataMatrix = mat(X)
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zeros((m, 1)))
        for i in range(len(self.__weakClassArr)):
            classEst = stumpClassify(dataMatrix, self.__weakClassArr[i][label_id]['dim'],
                                     self.__weakClassArr[i][label_id]['thresh'],
                                     self.__weakClassArr[i][label_id]['ineq'])
            aggClassEst += self.__weakClassArr[i][label_id]['alpha'] * classEst
#

        return sign(aggClassEst)
