#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hadaboost_mh import *
from label_trans import *
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle

########################Input Data##############################

train_x, label_train = data_process("ImCLEF07A_Train.arff")
test_x,  label_test = data_process("ImCLEF07A_Test.arff")       

########################label Preparation########################
train_label_count = label_count(label_train)
test_label_count = label_count(label_test)
test_y = []
for i in range(len(label_test)):
    test_y.append(label_test[i][0])
pickle.dump(test_y  ,open('test_label_07A.txt', 'wb') ) 
   
##################Hierarchical Structure########################
name = []
num_label_hier  = []
for x in range(len(train_label_count)):
    hier_num = x
    train_y = label_separate(label_train, hier_num)
    
    class_num = len(train_label_count[hier_num])
    class_name = train_label_count[hier_num]
    name.append(class_name)
    num_label_hier.append(len(class_name))
    locals()['train_y_mat_'+str(x)] = mat(zeros((len(train_y), class_num)))   
    for j in range(len(train_y)):
        for i in range(class_num):
#            if train_y[j][0] == class_name[i]:
            if train_y[j] == class_name[i]:
                locals()['train_y_mat_'+str(x)][j, i] = 1
            else:
                locals()['train_y_mat_'+str(x)][j, i] = -1

train_y_mat= None
for y in range(len(train_label_count) - 1):
    if train_y_mat  is None:
        train_y_mat = hstack((locals()['train_y_mat_'+str(y)],locals()['train_y_mat_'+str(y+1)]))
    else:
        train_y_mat = hstack((train_y_mat,locals()['train_y_mat_'+str(y+1)]))

class_num = shape(train_y_mat)[1]###
classes_name = [] 

for i in range(len(name)):
    for j in range(len(name[i])):
        classes_name.append(name[i][j])
        
##################Hierarchical Label Weight Assignment########################       
hier_weight = []
sum_hier_weight = 1
for i in range(len(num_label_hier)):
    if (i == 0):
        hier_weight.append(1/num_label_hier[i])
        sum_hier_weight = sum_hier_weight + 1
    else:
        hier_weight.append(hier_weight[i-1]/num_label_hier[i])
        sum_hier_weight = sum_hier_weight + hier_weight[i-1]

weighted_hier = []
for i in range(len(name)):
    for j in range(len(name[i])):
        weighted_hier.append(hier_weight[i]/sum_hier_weight)
 
##################Training Process###############################  
num_iter = 600
model = HAdaBoostMH(class_num, num_iter, weighted_hier)
weakClassArr, aggClassEst = model.fit(train_x, train_y_mat, name)

##################Training Paramenter Store########################  
pickle.dump(weakClassArr ,open('hier_para_07A.txt', 'wb') ) 
pickle.dump(name ,open('hier_name_07A.txt', 'wb') ) 


##################Prediction Process#############################
##predictlist = []
##label_num = 0
##for i in range(len(name)):
##    predict_list = []
##    for j in range(len(name[i])):
##        predict = model.predict(test_x, label_num)
##        label_num = label_num + 1
##        predict_list.append(predict)
##    predictlist.append(predict_list)
#######
##predict_final = []
##for i in range(len(name)):
##    predict_sum = []    
##    for j in range(len(test_x)):  
##        check_list = predictlist[i]
##        l = [x[j] for x in check_list]
##        class_name = name[i]
##        predict_sum.append(class_name[l.index(max(l))])
##    predict_final.append(predict_sum)
###    
####predictlist = []
####for j in range(len(name)):
####for i in range(len(classes_name)):
####    predict = model.predict(test_x, i)
####    predictlist.append(predict)
####
####predict_sum = []
####for i in range(len(test_x)):
####    l = [x[i] for x in predictlist]
####    predict_sum.append(classes_name[l.index(max(l))])
##print(metrics.classification_report(test_y, predict_final[0]))
####
##image_lists=pickle.load(open('Dataset_hier_07A.txt', 'rb'))
