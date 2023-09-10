import copy

import numpy as np


class Graph:
    def __init__(self, data_path, fea_dim):
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
        self.improve_mask = np.ones((self.size, self.size)) * (-np.inf)
        self.finsh_node = 1

    def generate(self, action):
        target_node = action
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
        for node in self.solution[:-1]:
            self.improve_mask[node, self.finsh_node] = 0
        self.improve_mask[target_node, 0:self.finsh_node] = 0
        self.improve_mask[target_node, self.finsh_node] = -np.inf
        self.finsh_node += 1

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
        state_adj_matrix = np.zeros((self.size, self.size))
        before_node = self.solution[0]
        for after_node in self.solution[1:]:
            state_adj_matrix[before_node][after_node] = 1
            state_adj_matrix[self.solution[-1]][before_node] = -1
            before_node = after_node
        state_adj_matrix[self.solution[-1]][before_node] = -1
        state_adj_matrix[self.solution[-1],:] += 1


        return self.node_fea, state_adj_matrix,self.generate_mask, self.improve_mask,

    def get_makespan(self, solution=0):
        if solution == 0:
            solution = self.solution
        distance = 0
        for node in solution:
            distance += self.node_fea[node, 3]
        return distance


if __name__ == "__main__":
    file_path = './dataset/20/tsp_data_100_20_2_0000.npy'
    graph = Graph(file_path, 4)
    graph.generate(4)
    graph.generate(6)
    graph.generate(2)
    graph.generate(12)
    graph.generate(8)
    graph.improve(85)
    graph.get_fea()
    print('OK')
