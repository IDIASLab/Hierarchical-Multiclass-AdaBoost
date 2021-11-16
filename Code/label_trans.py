#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:41:27 2020

@author: qwt
"""

import re
from pandas.core.frame import DataFrame as df

def data_process(filename):
    f = open(filename)
    lines = f.readlines()
    content = []
    for l in lines:
        content.append(l)    
    # filter the raw data without the attribution notation
    raw_data = []
    for i in range(len(content)):
        if "," in content[i]:
            raw_data.append(content[i])           
    data = []
    label = []
    for j in range(1, len(raw_data)):
        data_line = raw_data[j]
        data_split = data_line.split(",")
        list_new = [int(x) for x in data_split[:-1]]
        data.append(list_new)
        label_hierarchy = data_split[-1]
        label_hier = label_hierarchy.split("@")
        hier = re.split('/|\n', label_hier[-1])
        label.append(hier[:-1])
    return data, label

def label_count(label_set):
    label_count = {}
    label = df(label_set)
    for i in range(label.shape[1]):
        line = set(label[i])
        label_count[i] = list(line)
#        label_count[i].sort()
    return label_count

def label_separate(label_set, hier_index):
    label = df(label_set)
    line = list(label[hier_index])
    return line

#x_data_ori = pd.read_csv('data.csv', sep = ',' )
#c_label_ori = pd.read_csv('label_child.csv', sep = ',')
#p_label_ori = pd.read_csv('label_parent.csv', sep = ',')
#
#data_original = pd.concat([x_data_ori, c_label_ori, p_label_ori], axis = 1)
#data_shuffle = shuffle(data_original)
##raw data column index range:
#x_data = data_shuffle.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
#data_with_child = data_shuffle.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
##child_label column index range:
#p_label = data_shuffle.iloc[:,[12]]
#
#x_train, x_test, y_train, y_test = train_test_split(x_data , p_label, test_size = 0.3)




#def row2dict(row):
#    dict = []
#    for i in range(1, 19):
#        dict.append((float(row[i])))
#    return dict
#
#
#def loadData(data, label):
#    # 加载数据
#    reader = csv.reader(open(trainfilename, 'r'))
#    train_x = []
#    train_y = []
#    for row in islice(reader, 5, None):
#        train_x.append(row2dict(row))
#        train_y.append(row[0])
#    print 'load %d train_data complete!' % (len(train_x))
#    # 加载测试集
#    reader = csv.reader(open(testfilename, 'r'))
#    test_x = []
#    test_y = []
#    for row in islice(reader, 5, None):
#        test_x.append(row2dict(row))
#        test_y.append(row[0])
#    print 'load %d test_data complete!' % (len(test_x))
#    return train_x, train_y, test_x, test_y