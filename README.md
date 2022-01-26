# Hierarchical-Multiclass-AdaBoost

We present an extention of the AdaBoost algorithm for hierarchical multiclass classification.

## Citation
To cite our paper, please use the following reference:

Charalampos Chelmis, Wenting Qi. "Hierarchical MultiClass AdaBoost." 2021 IEEE International Conference on Big Data (Big Data), 2021. doi: 10.1109/BigData52589.2021.9671291.

BibTeX:
``` 
@article{chelmis2021hada, 
  author={Chelmis, Charalampos and Qi, Wenting},
  booktitle={2021 IEEE International Conference on Big Data (Big Data)}, 
  title={Hierarchical MultiClass AdaBoost}, 
  year={2021},
  volume={},
  number={},
  pages={5063-5070},
  doi={10.1109/BigData52589.2021.9671291}
  }
```

### Prerequisites
Python 3.6 or above and the following libraries
```
numpy
pickle
pandas
sklearn
```

## Files
``` 
  ImCLEF07A_Test.arff: a sample labeled test dataset
  ImCLEF07A_Train.arff: a sample labeled train dataset
  run_hada_mh.py: main running file of the H-Adaboost.MH
  label_trans.py: hierarchical label transerfer function file; called by run_hada_mh.py
  hadaboost_mh.py: main class of the H-Adaboost.MH; called by run_hada_mh.py
  decision_stump.py: weak classifier function file; called by run_hada_mh.py
```

### How to use
```
Step 1. Make sure run_hada_mh.py; label_trans.py; hadaboost_mh.py; decision_stump.py under same path

Step 2. Open the run_hada_mh.py file

Step 3. Set the path of train/test file in the "Input data section"
        Example: train_x, label_train = data_process("ImCLEF07A_Train.arff")
                 test_x,  label_test = data_process("ImCLEF07A_Test.arff")    

Step 4. Set the a appropraite number of weak classifiers in the "Training Process"
         Example: num_iter = 600

Step 5. Running the run_hada_mh.py file
```
