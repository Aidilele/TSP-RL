import numpy as np
import random
from copy import deepcopy


class ReplayBuffer:
    def __init__(self, buffer_size=2048, batch_size=256, ope_num=0, machine_num=0, gat_output_dim=0):

        ope_dim = 4
        self.ope_fea = np.zeros([buffer_size, ope_num, ope_dim])
        self.ope_adj = np.zeros([buffer_size, ope_num, ope_num])
        self.action_mask = np.zeros([buffer_size, ope_num])
        self.last_node = np.zeros([buffer_size, ope_num])
        self.h = np.zeros([buffer_size, gat_output_dim*2])
        self.c = np.zeros([buffer_size, gat_output_dim*2])
        self.action = np.zeros([buffer_size])
        self.reward = np.zeros([buffer_size])

        self.next_ope_fea = np.zeros([buffer_size, ope_num, ope_dim])
        self.next_ope_adj = np.zeros([buffer_size, ope_num, ope_num])
        self.next_action_mask = np.zeros([buffer_size, ope_num])
        self.next_last_node = np.zeros([buffer_size, ope_num])
        self.hn = np.zeros([buffer_size, gat_output_dim*2])
        self.cn = np.zeros([buffer_size, gat_output_dim*2])
        self.done = np.zeros([buffer_size])
        self.count = 0
        self.total_count = 0
        self.max_size = buffer_size
        self.batch_size = batch_size

    def store(self, ope_fea, ope_adj, action_mask, last_node,h, c,
              action,
              reward,
              next_ope_fea, next_ope_adj, next_action_mask,next_last_node, hn, cn,
              done,
              ep_size):
        start = self.count
        end = (self.count + ep_size) % self.max_size
        self.total_count += ep_size
        self.count = end
        if end > start:
            self.ope_fea[start:end] = ope_fea
            self.ope_adj[start:end] = ope_adj
            self.action_mask[start:end] = action_mask
            self.last_node[start:end] = last_node
            self.h[start:end] = h
            self.c[start:end] = c
            self.action[start:end] = action
            self.reward[start:end] = reward
            self.next_ope_fea[start:end] = next_ope_fea
            self.next_ope_adj[start:end] = next_ope_adj
            self.next_action_mask[start:end] = next_action_mask
            self.next_last_node[start:end] = next_last_node
            self.hn[start:end] = hn
            self.cn[start:end] = cn
            self.done[start:end] = done
        elif end < start:
            seg_ment = self.max_size - start
            self.ope_fea[start:] = ope_fea[:seg_ment]
            self.ope_adj[start:] = ope_adj[:seg_ment]

            self.action_mask[start:] = action_mask[:seg_ment]
            self.last_node[start:] = last_node[:seg_ment]
            self.h[start:] = h[:seg_ment]
            self.c[start:] = c[:seg_ment]
            self.action[start:] = action[:seg_ment]
            self.reward[start:] = reward[:seg_ment]
            self.next_ope_fea[start:] = next_ope_fea[:seg_ment]
            self.next_ope_adj[start:] = next_ope_adj[:seg_ment]

            self.next_action_mask[start:] = next_action_mask[:seg_ment]
            self.next_last_node[start:] = next_last_node[:seg_ment]
            self.hn[start:] = hn[:seg_ment]
            self.cn[start:] = cn[:seg_ment]
            self.done[start:] = done[:seg_ment]

            self.ope_fea[:end] = ope_fea[seg_ment:]
            self.ope_adj[:end] = ope_adj[seg_ment:]

            self.action_mask[:end] = action_mask[seg_ment:]
            self.last_node[:end] = last_node[seg_ment:]
            self.h[:end] = c[seg_ment:]
            self.c[:end] = c[seg_ment:]
            self.action[:end] = action[seg_ment:]
            self.reward[:end] = reward[seg_ment:]
            self.next_ope_fea[:end] = next_ope_fea[seg_ment:]
            self.next_ope_adj[:end] = next_ope_adj[seg_ment:]

            self.next_action_mask[:end] = next_action_mask[seg_ment:]
            self.next_last_node[:end] = next_last_node[seg_ment:]
            self.hn[:end] = cn[seg_ment:]
            self.cn[:end] = cn[seg_ment:]
            self.done[:end] = done[seg_ment:]

    def sample(self):
        batch = np.random.choice(self.max_size, self.batch_size, replace=False)
        ope_fea = self.ope_fea[batch]
        ope_adj = self.ope_adj[batch]

        action_mask = self.action_mask[batch]
        last_node = self.last_node[batch]
        h = self.h[batch]
        c = self.c[batch]
        action = self.action[batch]
        reward = self.reward[batch]
        next_ope_fea = self.next_ope_fea[batch]
        next_ope_adj = self.next_ope_adj[batch]

        next_action_mask = self.next_action_mask[batch]
        next_last_node = self.next_last_node[batch]

        hn = self.hn[batch]
        cn = self.cn[batch]
        done = self.done[batch]

        return (ope_fea, ope_adj, action_mask,last_node,
                h, c,
                action,
                reward,
                next_ope_fea, next_ope_adj, next_action_mask,next_last_node,
                hn, cn,
                done)
