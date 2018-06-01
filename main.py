import numpy as np
from tree import RT
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def data_loading(file_name):
    # read data from a file
    with open(file_name) as r:
        reader = csv.reader(r)
        data = [element for element in reader]
    return np.array(data)

def data_partition(input_data):
    # split the data into training and testing data equally
    header = input_data[0]
    tmp = np.delete(input_data, (0), axis = 0)
    np_list =  np.vsplit(tmp, 4)
    
    # generate training data and testing data
    final_data = np.concatenate((np.concatenate((np_list[0], np_list[1]), axis = 0), np_list[2]), axis = 0)
    return header, final_data, np_list[3]

def validation_data_partition(the_input, k, choice):
    np_list2 = np.vsplit(the_input, k)
    
    # create an index list for training and validation data creation
    temp = list()
    for i in range(k):
        if i != (choice - 1):
            temp.append(np_list2[i])
    
    return np.concatenate(tuple(temp), axis = 0), np_list2[choice - 1]

def data_processing(in_data):
    numerical_data = np.zeros(in_data.shape)
    
    # phase 1: convert all elements to numerical resulrs
    for row_index in range(in_data.shape[0]):
        for col_index in range(in_data.shape[1]):
            if in_data[row_index, col_index] == 'Yes':
                numerical_data[row_index, col_index] = 1
            elif in_data[row_index, col_index] == 'No':
                numerical_data[row_index, col_index] = -1
            elif in_data[row_index, col_index] == 'Bad':
                numerical_data[row_index, col_index] = 1   
            elif in_data[row_index, col_index] == 'Medium':
                numerical_data[row_index, col_index] = 3
            elif in_data[row_index, col_index] == 'Good':
                numerical_data[row_index, col_index] = 5
            else:
                numerical_data[row_index, col_index] = float(in_data[row_index, col_index])
    return numerical_data

def testing_phase(input_class, input_tree, data):
    mean_square_error = 0
    
    # loop through each testing instance
    for new_instance in data:
        predict_value = input_class.instance_prediction(input_tree, new_instance)
        # accumulate squar errors
        mean_square_error += (new_instance[0] - predict_value)**2
    # =======================================================
    # calculate MSE
    return mean_square_error / float(data.shape[0])

def random_forests(tree_num, in_features, input_data, depths,
                   min_left_group_size, min_right_group_size, in_tests):
    average_mse = 0
    average_training_error = 0
    feature_num = int(np.ceil(np.sqrt(len(in_features) - 1)) + 1)
    choice_list = list(range(1, len(in_features)))
    tree_root_list = []
    for tree_ind in range(tree_num):
        random_data = np.zeros((input_data.shape[0], feature_num + 1))
        random_data_test = np.zeros((in_tests.shape[0], feature_num + 1))
        random_feature = []
        indices = np.random.choice(choice_list, feature_num, replace = True)
        # create a new training data and a new feature list
        random_data[:, 0] = input_data[:, 0]
        random_data_test[:, 0] = in_tests[:, 0]
        random_feature.append(in_features[0])
        for col_ind in range(indices.shape[0]):
            random_data[:, col_ind + 1] = input_data[:, indices[col_ind]]
            random_data_test[:, col_ind + 1] = in_tests[:, indices[col_ind]]
            random_feature.append(in_features[indices[col_ind]])
        # build a regression tree
        tree_class = RT(np.asarray(random_feature), random_data)
        out_tree, leaves = tree_class.training_phase(depths, min_left_group_size, min_right_group_size)
        tree_root_list.append(tree_class.get_tree_root(out_tree, np.asarray(random_feature)))
        # compute training errors and the MSE values
        average_mse += testing_phase(tree_class, out_tree, random_data_test)
        average_training_error += testing_phase(tree_class, out_tree, random_data)
        final_mse = average_mse / float(tree_num)
        final_training_error = average_training_error / float(tree_num)
    return final_training_error, final_mse, tree_root_list

def regression_tree_bagging(max_epoch, in_features, input_data, depths,
                            min_left_group_size, min_right_group_size, in_tests):
    average_mse = 0
    average_training_error = 0
    tree_root_list = []
    for epoch_index in range(max_epoch):
        random_data = np.zeros_like(input_data, float)
        # randomly generate data indices
        indices = np.random.choice(random_data.shape[0], input_data.shape[0], replace = True)        

        # create a new training data
        for index in range(indices.shape[0]):
        #indices = np.random.randint(input_data.shape[0], size = random_data.shape[0])
            random_data[index, :] = input_data[indices[index], :]
        # build a regression tree
        tree_class = RT(in_features, random_data)
        out_tree, leaves = tree_class.training_phase(depths, min_left_group_size, min_right_group_size)
        tree_root_list.append(tree_class.get_tree_root(out_tree, in_features))
        # calculate average MSE
        average_mse += testing_phase(tree_class, out_tree, in_tests)
        average_training_error += testing_phase(tree_class, out_tree, random_data)
        final_mse = average_mse / float(max_epoch)
        final_training_error = average_training_error / float(max_epoch)
    return final_training_error, final_mse, tree_root_list

def regression_tree_pruning(in_features, input_data, depths, in_tests):
    min_group_size_list = list(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]))
    final_collec = []
    
    # go through each alpha value
    for group_size in list(min_group_size_list):
        k_folder_list= []
        # cross validation: go through each folder
        for i in range(len(min_group_size_list)):
            # validation data generation
            small_training, validation_data = validation_data_partition(input_data, len(min_group_size_list), i)
            # build the regression tree
            tree_class = RT(in_features, small_training)
            one_rt, leaves = tree_class.training_phase(depths, group_size, group_size)
            final_collec.append([group_size, one_rt, testing_phase(tree_class, one_rt, validation_data), i])
    
    # find the subtree
    mse_list = []
    for index in range(len(final_collec)):
        mse_list.append(final_collec[index][2])
    min_index = np.argmin(mse_list[1:]) + 1
    # calculate the new testing MSE
    small_training2, validation_data = validation_data_partition(input_data, len(min_group_size_list), final_collec[min_index][3])
    tree_class2 = RT(in_features, small_training2)
    tree_class2.display_tree(final_collec[min_index][1], depths, in_features)
    print('Testing MSE of Pruned Tree:',testing_phase(tree_class2, final_collec[min_index][1], in_tests))
    
def main():
    # variable setting
    name = 'Carseats.csv'
    min_left_group_size = 1
    min_right_group_size = 1
    initial_depth = 10
    max_bagging_epoch = 2000
    max_tree_num_forests = 2000

    # Question 1: Data partition and data preprocessing:
    # perform data partition to yield training data and testing data
    features, training_data, testing_data = data_partition(data_loading(name))
    
    # data preprocessing
    new_training_data = data_processing(training_data)
    new_testing_data = data_processing(testing_data)

    # Question 2 to Question 5
    if (sys.argv[1] == 'create-regression-tree'):
        # Question 2: build a regression tree
        # create single tree and display it
        tree = RT(features, new_training_data)
        single_tree_depth = 8
        single_tree, tree_leaves = tree.training_phase(single_tree_depth, min_left_group_size, min_right_group_size)
        tree.display_tree(single_tree, 0, features)
        print('Training error of The Original Tree:', testing_phase(tree, single_tree, new_training_data))
        print('MSE for The Original Tree:',testing_phase(tree, single_tree, new_testing_data))

        # create multiple regression trees to see the relationship between the tree depth and MSE
        training_error_list = []
        mse_list = []
        for depth_value in range(1, 30):
            regression_tree, tree_leaves = tree.training_phase(depth_value, min_left_group_size, min_right_group_size)
            #tree.display_tree(regression_tree, 0, features)
            training_error_list.append(testing_phase(tree, regression_tree, new_training_data))
            mse_list.append(testing_phase(tree, regression_tree, new_testing_data))
        
        # draw the plot
        x = list(range(1, 30))
        fig, ax = plt.subplots()
        ax.plot(x, training_error_list, label = 'Training MSE')
        ax.plot(x, mse_list, label = 'Testing MSE')
        plt.title('Tree Depth verse Mean Square Error')
        plt.xlabel('Tree Depth')
        plt.ylabel('Mean Square Error (MSE)')
        legend = ax.legend(loc='upper left', shadow=True)
        fig.savefig('test.png')
        
    elif (sys.argv[1] == 'regression-tree-pruning'):
        # Question 3: tree pruning
        single_tree_depth = 8
        regression_tree_pruning(features, new_training_data, single_tree_depth, new_testing_data)
    elif (sys.argv[1] == 'bagging'):
        # Question 4: Bagging
        # base line MSE
        tree = RT(features, new_training_data)
        single_tree, tree_leaves = tree.training_phase(initial_depth, min_left_group_size, min_right_group_size)
        print('Training Error: (Tree Depth is 10)',testing_phase(tree, single_tree, new_training_data))
        print('MSE: (Tree Depth is 10)',testing_phase(tree, single_tree, new_testing_data))
        
        # write root information to a file
        with open("roots-bagging", 'w') as f:
            # Bagging
            out_average_training, out_average_testing, roots = regression_tree_bagging(max_bagging_epoch, 
                                                                                features, new_training_data,
                                                                                initial_depth, min_left_group_size, 
                                                                                min_right_group_size,
                                                                                new_testing_data)
            print('Tree Number:',max_bagging_epoch)
            print('Average Training Error (Bagging): (Tree Depth is 10)',out_average_training)
            print('Average Testing MSE (Bagging): (Tree Depth is 10)',out_average_testing)
            
            f.write('{}\n{}\n'.format('Tree Number: ' + str(max_bagging_epoch), 'Tree Roots: ' + str(roots)))
    
    elif (sys.argv[1] == 'random-forests'):
        # Question 5: Random forests
        # base line MSE
        tree = RT(features, new_training_data)
        single_tree, tree_leaves = tree.training_phase(initial_depth, min_left_group_size, min_right_group_size)
        print('Training Error: (Tree Depth is 10)',testing_phase(tree, single_tree, new_training_data))
        print('MSE: (Tree Depth is 10)',testing_phase(tree, single_tree, new_testing_data))

        average_training_error_list = []
        average_mse_list = []
        # write root information to a file
        with open("roots-random-forests", 'w') as f:
            # random forests
            out_avg_training, out_avg_testing, roots = random_forests(max_tree_num_forests, features, new_training_data,
                                                               initial_depth, min_left_group_size, min_right_group_size,
                                                               new_testing_data)
            
            print('Tree Number:',max_tree_num_forests)
            print('Average Training Error (Random Forests): (Tree Depth is 10)',out_avg_training)
            print('Average Testing MSE (Random Forests): (Tree Depth is 10)',out_avg_testing)
            f.write('{}\n{}\n'.format('Tree Number: ' + str(max_tree_num_forests), 'Tree Roots: ' + str(roots)))
            
    else:
        print('Error Input. Please re-choose the task!!')

if __name__ == "__main__":
    main()
