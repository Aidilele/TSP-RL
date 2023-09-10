import copy
import os
import numpy as np


class Graph:
    def __init__(self, data_path, fea_dim=4):
        data = np.load(data_path)
        self.node_matrix = data - data.mean(0)
        self.size = data.shape[0]
        self.fea_dim = fea_dim
        self.adj_matrix = np.ones((self.size, self.size))
        self.node_matrix = data
        self.node_fea = np.zeros((self.size, self.fea_dim))
        self.node_fea[:, :2] = self.node_matrix
        self.solution = [0]
        self.solution_adj_matrix = np.zeros((self.size, self.size))
        self.finsh_node = 1
        self.generate_mask = np.zeros(self.size)
        self.improve_mask = np.ones((self.size, self.size)) * (-np.inf)
        self.last_node = np.zeros(self.size)

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
        self.solution_adj_matrix = np.zeros((self.size, self.size))
        self.node_fea[:, :2] = self.node_matrix
        self.generate_mask = np.zeros(self.size)
        self.generate_mask[0] = -np.inf
        self.improve_mask = np.ones((self.size, self.size)) * (-np.inf)
        self.last_node[0] = 1
        self.finsh_node = 1

    def generate(self, action):

        target_node = action
        self.last_node = np.zeros(self.size)
        self.last_node[action] = 1
        last_finsh_node = self.solution[-1]
        fisrt_node = self.solution[0]
        distance = np.linalg.norm(self.node_matrix[fisrt_node] - self.node_matrix[target_node])
        self.solution_adj_matrix[target_node, fisrt_node] = 1
        self.solution_adj_matrix[fisrt_node, :] = 0
        self.node_fea[fisrt_node, 2] = distance
        self.node_fea[target_node, 3] = distance
        self.solution.append(target_node)  # add target node to solution
        self.solution_adj_matrix[last_finsh_node, :] = 0
        self.solution_adj_matrix[last_finsh_node, target_node] = 1  # 添加有向边
        distance = np.linalg.norm(self.node_matrix[last_finsh_node] - self.node_matrix[target_node])
        self.node_fea[last_finsh_node, 3] = distance
        self.node_fea[target_node, 2] = distance
        self.generate_mask[target_node] = -np.inf

        # improve_mask 有待改善
        for node in self.solution[:-1]:
            self.improve_mask[node, self.finsh_node] = 0
        self.improve_mask[target_node, 0:self.finsh_node] = 0
        self.improve_mask[target_node, self.finsh_node] = -np.inf
        self.finsh_node += 1

        if len(self.solution) == self.size:
            self.generate_mask[0] = 0

            self.finsh_node = self.size - 1

    def improve(self, action):
        target_node = action // self.size
        target_position = action % self.size
        pre_node = self.solution[target_position - 1]
        sub_node = self.solution[target_position]
        pt_dis = np.linalg.norm(self.node_matrix[pre_node] - self.node_matrix[target_node])
        ts_dis = np.linalg.norm(self.node_matrix[target_node] - self.node_matrix[sub_node])
        self.node_fea[target_node, 2] = pt_dis
        self.node_fea[target_node, 3] = ts_dis
        self.node_fea[pre_node, 3] = pt_dis
        self.node_fea[sub_node, 2] = ts_dis
        self.solution_adj_matrix[pre_node, :] = 0
        self.solution_adj_matrix[pre_node, target_node] = 1
        self.solution_adj_matrix[target_node, :] = 0
        self.solution_adj_matrix[target_node, sub_node] = 1
        target_node_pre_pos = self.solution.index(target_node)
        old_pre_node = self.solution[target_node_pre_pos - 1]
        old_sub_node = self.solution[target_node_pre_pos + 1]
        dis = np.linalg.norm(self.node_matrix[old_pre_node] - self.node_matrix[old_sub_node])
        self.node_fea[old_pre_node, 3] = dis
        self.node_fea[old_sub_node, 2] = dis
        self.solution_adj_matrix[old_pre_node, old_sub_node] = 1
        if target_node_pre_pos < target_position:
            target_position -= 1
        self.solution.remove(target_node)
        self.solution.insert(target_position, target_node)
        index = 0
        self.improve_mask = np.ones((self.size, self.size)) * (-np.inf)
        for node in self.solution:
            self.improve_mask[node, :self.finsh_node] = 0
            self.improve_mask[node, index] = -np.inf
            index += 1
        self.improve_mask = self.improve_mask.reshape(-1)

    def get_fea(self):
        state_adj_matrix = np.eye(self.size)
        before_node = self.solution[0]
        for after_node in self.solution[1:]:
            state_adj_matrix[before_node][after_node] = 1
            state_adj_matrix[self.solution[-1]][before_node] = -1
            before_node = after_node
        state_adj_matrix[self.solution[-1]][before_node] = -1
        state_adj_matrix[self.solution[-1], :] += 1

        if len(self.solution) == self.size:
            state_adj_matrix[before_node][0] = 1

        if len(self.solution) >= 2:
            pre_node = self.solution[-2]
            sub_node = self.solution[-1]
            reward = np.linalg.norm(self.node_matrix[pre_node] - self.node_matrix[sub_node])
        else:
            reward = 0

        state_adj_matrix = state_adj_matrix + state_adj_matrix.T - np.eye(self.size)

        return (np.expand_dims(self.node_fea, 0),
                np.expand_dims(state_adj_matrix, 0),
                np.expand_dims(self.generate_mask, 0),
                np.expand_dims(self.improve_mask, 0),
                reward,
                np.expand_dims(self.last_node, 0),
                )

    def get_makespan(self, solution=0):
        if solution == 0:
            solution = self.solution
        distance = 0
        for node in solution:
            distance += self.node_fea[node, 3]
        return distance


if __name__ == "__main__":
    dir_path = './dataset/100'
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
