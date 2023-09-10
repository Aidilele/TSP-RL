import numpy as np
import random
from copy import deepcopy


class ReplayBuffer:
    def __init__(self, buffer_size=2048, batch_size=256, ope_num=0, machine_num=0, gat_output_dim=0):

        ope_dim = 2
        improve_deepth=10

        self.node_fea = np.zeros([buffer_size, ope_num, ope_dim])
        self.static_adj = np.zeros([buffer_size, ope_num, ope_num])
        self.dynamic_adj = np.zeros([buffer_size, ope_num, ope_num])
        self.insert_adj = np.zeros([buffer_size, improve_deepth+1, improve_deepth+2])
        self.blank_node = np.zeros([buffer_size, ope_num])
        self.insert_node = np.zeros([buffer_size, ope_num])
        self.action_mask = np.zeros([buffer_size, improve_deepth*(improve_deepth+1)])

        self.h = np.zeros([buffer_size, gat_output_dim*2+ope_dim])
        self.c = np.zeros([buffer_size, gat_output_dim*2+ope_dim])
        self.action = np.zeros([buffer_size])
        self.reward = np.zeros([buffer_size])

        self.next_node_fea = np.zeros([buffer_size, ope_num, ope_dim])
        self.next_static_adj = np.zeros([buffer_size, ope_num, ope_num])
        self.next_dynamic_adj = np.zeros([buffer_size, ope_num, ope_num])
        self.next_insert_adj = np.zeros([buffer_size, improve_deepth+1, improve_deepth+2])
        self.next_blank_node = np.zeros([buffer_size, ope_num])
        self.next_insert_node = np.zeros([buffer_size, ope_num])
        self.next_action_mask = np.zeros([buffer_size, improve_deepth*(improve_deepth+1)])


        self.hn = np.zeros([buffer_size, gat_output_dim*2+ope_dim])
        self.cn = np.zeros([buffer_size, gat_output_dim*2+ope_dim])
        self.done = np.zeros([buffer_size])
        self.count = 0
        self.total_count = 0
        self.max_size = buffer_size
        self.batch_size = batch_size

    def store(self,
              node_fea, static_adj, dynamic_adj, insert_adj, blank_node, insert_node,
              action_mask,
              h, c,
              action,
              reward,
              next_node_fea, next_static_adj, next_dynamic_adj, next_insert_adj, next_blank_node, next_insert_node,
              next_action_mask,
              hn, cn,
              done,
              ep_size):
        start = self.count
        end = (self.count + ep_size) % self.max_size
        self.total_count += ep_size
        self.count = end
        if end > start:

            self.node_fea[start:end] = node_fea
            self.static_adj[start:end] = static_adj
            self.dynamic_adj[start:end] = dynamic_adj
            self.insert_adj[start:end] = insert_adj
            self.blank_node[start:end] = blank_node
            self.insert_node[start:end] = insert_node

            self.action_mask[start:end] = action_mask

            self.h[start:end] = h
            self.c[start:end] = c
            self.action[start:end] = action
            self.reward[start:end] = reward

            self.next_node_fea[start:end] = next_node_fea
            self.next_static_adj[start:end] = next_static_adj
            self.next_dynamic_adj[start:end] = next_dynamic_adj
            self.next_insert_adj[start:end] = next_insert_adj
            self.next_blank_node[start:end] = next_blank_node
            self.next_insert_node[start:end] = next_insert_node
            self.next_action_mask[start:end] = next_action_mask

            self.hn[start:end] = hn
            self.cn[start:end] = cn
            self.done[start:end] = done
        elif end < start:
            seg_ment = self.max_size - start

            self.node_fea[start:] = node_fea[:seg_ment]
            self.static_adj[start:] = static_adj[:seg_ment]
            self.dynamic_adj[start:] = dynamic_adj[:seg_ment]
            self.insert_adj[start:] = insert_adj[:seg_ment]
            self.blank_node[start:] = blank_node[:seg_ment]
            self.insert_node[start:] = insert_node[:seg_ment]
            self.action_mask[start:] = action_mask[:seg_ment]

            self.h[start:] = h[:seg_ment]
            self.c[start:] = c[:seg_ment]
            self.action[start:] = action[:seg_ment]
            self.reward[start:] = reward[:seg_ment]

            self.next_node_fea[start:] = next_node_fea[:seg_ment]
            self.next_static_adj[start:] = next_static_adj[:seg_ment]
            self.next_dynamic_adj[start:] = next_dynamic_adj[:seg_ment]
            self.next_insert_adj[start:] = next_insert_adj[:seg_ment]
            self.next_blank_node[start:] = next_blank_node[:seg_ment]
            self.next_insert_node[start:] = next_insert_node[:seg_ment]
            self.next_action_mask[start:] = next_action_mask[:seg_ment]

            self.hn[start:] = hn[:seg_ment]
            self.cn[start:] = cn[:seg_ment]
            self.done[start:] = done[:seg_ment]

            self.node_fea[:end] = node_fea[seg_ment:]
            self.static_adj[:end] = static_adj[seg_ment:]
            self.dynamic_adj[:end] = dynamic_adj[seg_ment:]
            self.insert_adj[:end] = insert_adj[seg_ment:]
            self.blank_node[:end] = blank_node[seg_ment:]
            self.insert_node[:end] = insert_node[seg_ment:]
            self.action_mask[:end] = action_mask[seg_ment:]

            self.h[:end] = c[seg_ment:]
            self.c[:end] = c[seg_ment:]
            self.action[:end] = action[seg_ment:]
            self.reward[:end] = reward[seg_ment:]

            self.next_node_fea[:end] = next_node_fea[seg_ment:]
            self.next_static_adj[:end] = next_static_adj[seg_ment:]
            self.next_dynamic_adj[:end] = next_dynamic_adj[seg_ment:]
            self.next_insert_adj[:end] = next_insert_adj[seg_ment:]
            self.next_blank_node[:end] = next_blank_node[seg_ment:]
            self.next_insert_node[:end] = next_insert_node[seg_ment:]
            self.next_action_mask[:end] = next_action_mask[seg_ment:]


            self.hn[:end] = cn[seg_ment:]
            self.cn[:end] = cn[seg_ment:]
            self.done[:end] = done[seg_ment:]

    def sample(self):
        batch = np.random.choice(self.max_size, self.batch_size, replace=False)
        node_fea = self.node_fea[batch]
        static_adj = self.static_adj[batch]
        dynamic_adj = self.dynamic_adj[batch]
        insert_adj = self.insert_adj[batch]
        blank_node = self.blank_node[batch]
        insert_node = self.insert_node[batch]
        action_mask = self.action_mask[batch]


        h = self.h[batch]
        c = self.c[batch]
        action = self.action[batch]
        reward = self.reward[batch]

        next_node_fea = self.next_node_fea[batch]
        next_static_adj = self.next_static_adj[batch]
        next_dynamic_adj = self.next_dynamic_adj[batch]
        next_insert_adj = self.next_insert_adj[batch]
        next_blank_node = self.next_blank_node[batch]
        next_insert_node = self.next_insert_node[batch]
        next_action_mask = self.next_action_mask[batch]

        hn = self.hn[batch]
        cn = self.cn[batch]
        done = self.done[batch]

        return (node_fea, static_adj, dynamic_adj, insert_adj, blank_node, insert_node,
                action_mask,
                h, c,
                action,
                reward,
                next_node_fea, next_static_adj, next_dynamic_adj, next_insert_adj, next_blank_node, next_insert_node,
                next_action_mask,
                hn, cn,
                done)
