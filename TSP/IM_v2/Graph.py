import copy
import os
import numpy as np


class Graph:
    def __init__(self, data_path, fea_dim=2, improve_deepth=10):
        data = np.load(data_path)
        self.node_matrix = data - data.mean(0)
        self.size = data.shape[0]
        self.fea_dim = fea_dim
        self.improve_deepth = improve_deepth
        self.adj_matrix = np.ones((self.size, self.size))
        self.node_matrix = data
        self.node_fea = np.zeros((self.size, self.fea_dim))
        self.node_fea[:, :2] = self.node_matrix
        self.solution = [0]
        self.solution_adj_matrix = np.eye(self.size)
        self.finsh_node = 1
        self.generate_mask = np.zeros(self.size)
        self.improve_mask = np.zeros((self.improve_deepth, self.improve_deepth + 1))
        inf_eye = np.eye(self.improve_deepth)
        for i in range(self.improve_deepth):
            inf_eye[i, i] = -np.inf
        self.improve_mask[:, :self.improve_deepth] += inf_eye
        self.improve_mask[:, 1:self.improve_deepth + 1] += inf_eye
        self.improve_mask = self.improve_mask.reshape(self.improve_deepth * (self.improve_deepth + 1))
        self.last_node = np.zeros(self.size)
        self.insert_node = np.zeros(self.size)
        self.blank_node = np.zeros(self.size)
        self.insert_adj_matrix = np.zeros((self.improve_deepth + 1, self.improve_deepth + 2))
        self.insert_adj_matrix[:, :self.improve_deepth + 1] += np.eye(self.improve_deepth + 1)
        self.insert_adj_matrix[:, 1:self.improve_deepth + 2] += np.eye(self.improve_deepth + 1)

    def reset(self):
        '''
        node_fea:
            0.x
            1.y
            2.pre_city_distance
            3.sub_city_distance
        '''
        self.solution = [0]
        self.node_fea = np.zeros((self.size, self.fea_dim))
        self.solution_adj_matrix = np.eye(self.size)
        self.solution_adj_matrix[0, 1:] = 1
        self.solution_adj_matrix[1:, 0] = 1
        self.node_fea[:, :2] = self.node_matrix
        self.generate_mask = np.zeros(self.size)
        self.generate_mask[0] = -np.inf
        self.last_node[0] = 1
        self.finsh_node = 1
        self.insert_node[1:self.improve_deepth + 1] = 1
        self.blank_node[0:self.improve_deepth + 2] = 1
        self.slider_start = 0
        self.slider_end = self.improve_deepth + 2

    def generate_action(self):
        if self.finsh_node == self.size:
            return 0
        current_node = self.node_fea[self.finsh_node - 1]
        rest_node = self.node_fea[self.finsh_node:]
        max_index = np.argmin(np.linalg.norm((current_node - rest_node), axis=1))
        action = max_index + self.finsh_node
        return action

    def slider_window(self):
        self.slider_start += 1
        self.slider_end += 1
        if self.slider_end >= self.size+1:
            self.slider_start -= 1
            self.slider_end -= 1
            target_fea = self.node_fea[0].copy()
            self.node_fea[:self.size-1] = self.node_fea[1:self.size]
            self.node_fea[self.size-1] = target_fea



        self.blank_node[:] = 0
        self.blank_node[self.slider_start:self.slider_end] = 1
        self.insert_node[:] = 0
        self.insert_node[self.slider_start + 1:self.slider_end - 1] = 1

        return 0

    def generate(self, action):
        target_node = action
        target_fea = self.node_fea[target_node].copy()
        if action != 0:
            self.node_fea[self.finsh_node + 1:target_node + 1] = self.node_fea[self.finsh_node:target_node]
            self.node_fea[self.finsh_node] = target_fea
            before_node = self.finsh_node - 1
            after_node = self.finsh_node
            self.solution_adj_matrix[before_node, after_node:] = 0
            self.solution_adj_matrix[after_node:, before_node] = 0
            self.solution_adj_matrix[before_node, after_node] = 1
            self.solution_adj_matrix[after_node, before_node] = 1
            self.solution_adj_matrix[after_node, after_node:] = 1
            self.solution_adj_matrix[after_node:, after_node] = 1
            self.generate_mask[self.finsh_node] = -np.inf
        else:
            # self.node_fea[:self.finsh_node - 1] = self.node_fea[1:self.finsh_node]
            # self.node_fea[self.finsh_node - 1] = target_fea
            before_node = self.finsh_node - 1
            after_node = 0
            self.solution_adj_matrix[before_node, after_node] = 1
            self.solution_adj_matrix[after_node, before_node] = 1
            self.generate_mask[0] = -np.inf

        self.solution.append(action)
        self.finsh_node += 1
        if self.finsh_node == self.size:
            self.generate_mask[0] = 0
        if self.finsh_node >= self.improve_deepth + 2:
            self.insert_node[:] = 0
            self.blank_node[:] = 0
            start = min(self.finsh_node - self.improve_deepth - 1, self.size - self.improve_deepth - 1)
            end = min(self.finsh_node - 1, self.size - 1)

            self.insert_node[start:end] = 1
            self.blank_node[start - 1:end + 1] = 1

        return 0

    def improve(self, action):
        target_node = action // (self.improve_deepth + 1)
        target_position = action % (self.improve_deepth + 1)

        insert_segment = self.node_fea[self.finsh_node - 1 - self.improve_deepth:self.finsh_node].copy()
        target_node_fea = insert_segment[target_node].copy()
        if target_node < target_position:
            insert_segment[target_node:target_position - 1] = insert_segment[target_node + 1:target_position]
            insert_segment[target_position - 1] = target_node_fea
        else:
            try:
                insert_segment[target_position + 1:target_node] = insert_segment[target_position:target_node - 1]
                insert_segment[target_position] = target_node_fea
            except:
                pass

        self.node_fea[self.finsh_node - 1 - self.improve_deepth:self.finsh_node] = insert_segment

    def get_generate_fea(self):
        return (np.expand_dims(self.node_fea, 0),
                np.expand_dims(self.adj_matrix, 0),
                np.expand_dims(self.solution_adj_matrix, 0),
                np.expand_dims(self.generate_mask, 0),
                np.expand_dims(self.last_node, 0),
                )

    def get_improve_fea(self):
        return (np.expand_dims(self.node_fea, 0),
                np.expand_dims(self.adj_matrix, 0),
                np.expand_dims(self.solution_adj_matrix, 0),
                np.expand_dims(self.insert_adj_matrix, 0),
                np.expand_dims(self.blank_node, 0),
                np.expand_dims(self.insert_node, 0),
                np.expand_dims(self.improve_mask, 0),
                self.get_makespan()
                )

    def get_makespan(self):
        end = min(self.finsh_node, self.size)
        segment1 = self.node_fea[:end].copy()
        segment2 = self.node_fea[:end].copy()
        last_node = segment1[0].copy()
        segment2[0:end - 1] = segment1[1:end]
        segment2[end - 1] = last_node
        mines = segment1 - segment2
        distance = np.linalg.norm(mines, axis=1).sum()

        return distance


if __name__ == "__main__":
    # dir_path = './dataset/20'
    # graph_list = []
    # for file in os.listdir(dir_path):
    #     file_path = dir_path + '/' + file
    #     graph = Graph(file_path)
    #     graph_list.append(graph)
    #
    # total_gap = 0
    # node_num = graph_list[0].size
    # graph_list[0].reset()
    # for i in range(graph_list[0].size):
    #     graph_list[0].generate(i + 1)
    # print(0)

    dir_path = './dataset/20test'
    graph_list = []
    for file in os.listdir(dir_path):
        file_path = dir_path + '/' + file
        graph = Graph(file_path)
        graph_list.append(graph)

    total_gap = 0
    node_num = graph_list[0].size
    for graph in graph_list:
        range_matrix = np.zeros((node_num, node_num))
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    range_matrix[i][j] = 1
                else:
                    range_matrix[i][j] = np.linalg.norm(graph.node_matrix[i] - graph.node_matrix[j])
        solution = [0]
        current_node = 0
        range_matrix[:, current_node] = 1
        next_node = 0
        sum_distance = 0
        for i in range(node_num - 1):
            next_node = np.argmin(range_matrix[current_node])
            range_matrix[:, current_node] = 1
            distance = range_matrix[current_node][next_node]
            sum_distance += distance
            current_node = next_node
        sum_distance += range_matrix[0][current_node]
        print(sum_distance)
        total_gap += sum_distance

    print(total_gap / len(graph_list))
