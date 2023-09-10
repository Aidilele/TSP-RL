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
        self.last_node = np.zeros(self.size)
        self.insert_node = np.zeros(self.size)
        self.blank_node = np.zeros(self.size)
        self.insert_adj_matrix = np.zeros((self.improve_deepth + 1, self.improve_deepth + 2))

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
        self.solution_adj_matrix[1:, 0] = -1
        self.node_fea[:, :2] = self.node_matrix
        self.generate_mask = np.zeros(self.size)
        self.generate_mask[0] = -np.inf
        self.last_node[0] = 1
        self.finsh_node = 1

        inf_eye = np.eye(self.improve_deepth)
        for i in range(self.improve_deepth):
            inf_eye[i, i] = -np.inf
        self.improve_mask[:, :self.improve_deepth] += inf_eye
        self.improve_mask[:, 1:self.improve_deepth + 1] += inf_eye
        self.improve_mask.reshape(self.improve_deepth * (self.improve_deepth + 1))

        self.insert_adj_matrix[:, :self.improve_deepth + 1] += np.eye(self.improve_deepth + 1)
        self.insert_adj_matrix[:, 1:self.improve_deepth + 2] += np.eye(self.improve_deepth + 1)

        self.insert_node[1:self.improve_deepth + 1] = 1
        self.blank_node[0:self.improve_deepth + 2] = 1

    def generate(self, action):

        target_node = action
        target_fea = self.node_fea[target_node]
        self.node_fea[self.finsh_node + 1:target_node + 1] = self.node_fea[self.finsh_node:target_node]
        self.node_fea[self.finsh_node] = target_fea

        before_node = self.finsh_node - 1
        after_node = target_node
        self.solution.append(action)

        self.solution_adj_matrix[before_node, after_node + 1:] = 0
        self.solution_adj_matrix[after_node + 1:, before_node] = 0
        self.solution_adj_matrix[after_node, after_node - 1] = 1

        self.solution_adj_matrix[after_node, after_node:] = 1
        self.solution_adj_matrix[after_node + 1:, after_node] = 1
        self.generate_mask[after_node] = -np.inf
        self.finsh_node += 1
        if self.finsh_node == self.size:
            self.generate_mask[0] = 0
        if self.finsh_node >= self.improve_deepth + 2:
            self.insert_node[:] = 0
            self.blank_node[:] = 0
            self.insert_node[self.finsh_node - self.improve_deepth - 1:self.finsh_node - 1] = 1
            self.blank_node[self.finsh_node - self.improve_deepth - 2:self.finsh_node] = 1

    def improve(self, action):
        target_node = action // (self.improve_deepth + 1)
        target_position = action % (self.improve_deepth + 1)

        target_node_fea = self.node_fea[target_node]
        insert_segment = self.node_fea[self.finsh_node - 1 - self.improve_deepth:self.finsh_node - 1]
        insert_segment[target_position + 1:target_node + 1] = insert_segment[target_position:target_node]
        insert_segment[target_position] = target_node_fea
        self.node_fea[self.finsh_node - 1 - self.improve_deepth:self.finsh_node - 1] = insert_segment

        insert_segment_solution = self.solution[self.finsh_node - 1 - self.improve_deepth:self.finsh_node - 1]
        insert_segment_solution[target_position + 1:target_node + 1] = insert_segment_solution[
                                                                       target_position:target_node]
        insert_segment_solution[target_position] = target_node
        self.solution[self.finsh_node - 1 - self.improve_deepth:self.finsh_node - 1] = insert_segment_solution

    def get_fea(self):
        if len(self.solution) >= 2:
            pre_node = self.solution[-2]
            sub_node = self.solution[-1]
            reward = np.linalg.norm(self.node_matrix[pre_node] - self.node_matrix[sub_node])
        else:
            reward = 0

        return (np.expand_dims(self.node_fea, 0),
                np.expand_dims(self.adj_matrix, 0),
                np.expand_dims(self.solution_adj_matrix, 0),
                np.expand_dims(self.generate_mask, 0),
                np.expand_dims(self.improve_mask, 0),
                np.expand_dims(self.blank_node, 0),
                np.expand_dims(self.insert_node, 0),
                np.expand_dims(self.insert_adj_matrix, 0),
                np.expand_dims(self.last_node, 0),
                reward,
                )

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
                np.expand_dims(self.improve_mask, 0),
                np.expand_dims(self.blank_node, 0),
                np.expand_dims(self.insert_node, 0),
                np.expand_dims(self.insert_adj_matrix, 0),
                )

    def get_makespan(self, solution=0):
        if solution == 0:
            solution = self.solution
        distance = 0
        for node in solution:
            distance += self.node_fea[node, 3]
        return distance


if __name__ == "__main__":
    dir_path = './dataset/20'
    graph_list = []
    for file in os.listdir(dir_path):
        file_path = dir_path + '/' + file
        graph = Graph(file_path)
        graph_list.append(graph)

    total_gap = 0
    node_num = graph_list[0].size
    graph_list[0].reset()
    for i in range(graph_list[0].size):
        graph_list[0].generate(i + 1)
    print(0)

    # for graph in graph_list:
    #     range_matrix = np.zeros((node_num, node_num))
    #     for i in range(node_num):
    #         for j in range(node_num):
    #             if i == j:
    #                 range_matrix[i][j] = 1
    #             else:
    #                 range_matrix[i][j] = np.linalg.norm(graph.node_matrix[i] - graph.node_matrix[j])
    #     solution = [0]
    #     current_node = 0
    #     range_matrix[:, current_node] = 1
    #     next_node = 0
    #     sum_distance = 0
    #     for i in range(node_num):
    #         next_node = np.argmin(range_matrix[current_node])
    #         range_matrix[:, current_node] = 1
    #         distance = range_matrix[current_node][next_node]
    #         sum_distance += distance
    #         current_node = next_node
    #     # print(sum_distance)
    #     total_gap += sum_distance
    #     circle_distance = range_matrix[current_node][0]
    #     total_gap += circle_distance
    # print(total_gap / len(graph_list))
