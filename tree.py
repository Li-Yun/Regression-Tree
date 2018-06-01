import numpy as np
from statistics import median

# tree class
class RT:
    def __init__(self, in_feature, in_train):
        self.root = None
        self.collection = []
        self.feature_list = in_feature
        self.train_data = in_train
        self.tree_leaf = []
        self.tree_root = ''
    # ==============================
    def RSS_Calculation(self, input_data):
        record = []
        # walk through each feature
        for feature_ind in range(1, len(list(self.feature_list))):
            # remove duplicate elements in each feature
            split_point_list = list(set(input_data[:,feature_ind]))
            #split_point_list = list(np.arange(int(np.amin(self.train_data[:,feature_ind])), 
            #                                  int(np.amax(self.train_data[:,feature_ind])), 0.5))
            for split_point in split_point_list:
                R1_y_value = []
                R2_y_value = []
                for row_ind in range(input_data.shape[0]):
                    if self.train_data[row_ind, feature_ind] < float(split_point):
                        R1_y_value.append(input_data[row_ind, 0])
                    else:
                        R2_y_value.append(input_data[row_ind, 0])
                R1_mean = sum(R1_y_value) / float(len(R1_y_value) + 0.00001) 
                R2_mean = sum(R2_y_value) / float(len(R2_y_value) + 0.00001)

                # compute RSS given one feature index and a split point
                rss_value = np.sum(np.square(np.array(R1_y_value) - R1_mean)) + \
                            np.sum(np.square(np.array(R2_y_value) - R2_mean))
                record.append({'feature_index':feature_ind, 'split_point':split_point, 'rss':rss_value})
        return record
    # =====================================
    def Get_split_point(self, in_record):
        temp_list = []
        for tmp_ind in range(len(in_record)):
            temp_list.append(in_record[tmp_ind]['rss'])
        # find the best split point
        best_index = np.argmin(temp_list)
        return in_record[best_index]
    # =====================================
    def group_split(self, in_node, inputs):
        left_group = []
        right_group = []
        for data_ind in range(inputs.shape[0]):
            if inputs[data_ind, in_node['feature_index']] < in_node['split_point']:
                left_group.append(inputs[data_ind, :])
            else:
                right_group.append(inputs[data_ind, :])
        return np.array(left_group), np.array(right_group)
    # =====================================
    def create_terminal_node(self, input_group):
        # extract the Sales value
        outcomes = [row[0] for row in input_group]
        # return the median value
        return median(outcomes)
    # =====================================
    def display_tree(self, input_tree, tree_depth, input_feature):
        space_variable = ' '
        if isinstance(input_tree, dict):
            print(2*tree_depth*space_variable,'[',input_feature[input_tree['feature_index']]
                  ,' < ',input_tree['split_point'],': RSS',input_tree['rss'],']')
            self.display_tree(input_tree['left'], tree_depth + 1, input_feature)
            self.display_tree(input_tree['right'], tree_depth + 1, input_feature)
        else:
            print(2*tree_depth*space_variable,'[',input_tree,']')
    # =====================================
    def build_subtree(self, node, max_depth, tree_depth, inputs, min_left_group, min_right_group):
        # split the data into two groups
        left_data, right_data = self.group_split(node, inputs)
        #del(node['rss'])
        
        # check for a no split
        if left_data.size == 0 or right_data.size == 0:
            if left_data.size == 0:
                node['left'] = node['right'] = self.create_terminal_node(right_data)
                self.tree_leaf.append([node['left'], right_data])
                self.tree_leaf.append([node['right'], right_data])
            elif right_data.size == 0:
                node['left'] = node['right'] = self.create_terminal_node(left_data)
                self.tree_leaf.append([node['left'], left_data])
                self.tree_leaf.append([node['right'], left_data])
            return
        # check for max depth
        if tree_depth >= max_depth:
            node['left'] = self.create_terminal_node(left_data)
            node['right'] = self.create_terminal_node(right_data)
            self.tree_leaf.append([node['left'], left_data])
            self.tree_leaf.append([node['right'], right_data])
            return
        # process the left subtree
        if left_data.shape[0] <= min_left_group:
            node['left'] = self.create_terminal_node(left_data)
            self.tree_leaf.append([node['left'], left_data])
        else:
            node['left'] = self.Get_split_point(self.RSS_Calculation(left_data))
            self.build_subtree(node['left'], max_depth, tree_depth + 1, left_data, min_left_group, min_right_group)
        # process the right subtree
        if right_data.shape[0] <= min_right_group:
            node['right'] = self.create_terminal_node(right_data)
            self.tree_leaf.append([node['right'], right_data])
        else:
            node['right'] = self.Get_split_point(self.RSS_Calculation(right_data))
            self.build_subtree(node['right'], max_depth, tree_depth + 1, right_data, min_left_group, min_right_group)
    # =====================================
    # training stage: build the regression tree.
    def training_phase(self, max_depth, min_left_size, min_right_size):
        # get the split point and start with the root
        self.root = self.Get_split_point(self.RSS_Calculation(self.train_data))
        # build the subtree
        self.build_subtree(self.root, max_depth, 1, self.train_data, min_left_size, min_right_size)
        return self.root, self.tree_leaf
    # =====================================
    # get tree root
    def get_tree_root(self, input_tree, input_feature):
        if isinstance(input_tree, dict):
            self.tree_root = '[' + input_feature[input_tree['feature_index']] + ' < ' + str(input_tree['split_point']) + ': RSS ' + str(input_tree['rss']) + ']'
        return self.tree_root
    # =====================================
    # testing stage: prediction new instances
    def instance_prediction(self, input_tree, input_row):
        predict_value = 0
        # left subtree
        if input_row[input_tree['feature_index']] < input_tree['split_point']:
            if isinstance(input_tree['left'], dict):
                predict_value = self.instance_prediction(input_tree['left'], input_row)
            else:
                predict_value = input_tree['left']
        # right subtree
        else:
            if isinstance(input_tree['right'], dict):
                predict_value = self.instance_prediction(input_tree['right'], input_row)
            else:
                predict_value = input_tree['right']
        return predict_value

    
