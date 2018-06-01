# Regression-Tree

There are several regress tree implementations inclusing canonical regression tree, pruning tree, tree bagging, random forest. I implement my regression tree using Carseats dataset and create a pruned tree with small tree depth by applying Cross Validation (CV) to
choose an appropriate number of data points in one region. By doing this, the pruned tree can be yielded to get smaller Mean Square Error (MSE). For tree bagging, it is able to get the performance (small MSE values) by leveraging hundreds or thousands of regression trees. Finally, Random Forests is also implemented in this assignment to boost prediction performance of single regression tree.

References:
[1] https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
