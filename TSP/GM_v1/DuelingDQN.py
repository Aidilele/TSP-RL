import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import ReplayBuffer
from GATModel import *
from torch.utils.tensorboard import SummaryWriter

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class VNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, hidden_dim, hidden_layer):
        super(VNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                )
        for i in range(hidden_layer):
            self.fc.add_module('hidden_layer', nn.Linear(hidden_dim, hidden_dim))
            self.fc.add_module('activation_layer', nn.LeakyReLU())
        self.fc.add_module('output_layer', nn.Linear(hidden_dim, 1))
        self.to(device)

    def forward(self, state):
        output = self.fc(state)
        return output


class ANetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, hidden_dim, hidden_layer):
        super(ANetwork, self).__init__()
        # self.ope_net = OpeModel(ope_dim, gat_hidden_dim, gat_output_dim)
        self.fc = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                )
        for i in range(hidden_layer):
            self.fc.add_module('hidden_layer', nn.Linear(hidden_dim, hidden_dim))
            self.fc.add_module('activation_layer', nn.LeakyReLU())
        self.fc.add_module('output_layer', nn.Linear(hidden_dim, 1))
        self.to(device)
        self.to(device)

    def forward(self, state, action):
        action_num = action.shape[-2]
        state_dim = state.shape[-1]
        input = T.cat([state.repeat(1, action_num).view(-1, action_num, state_dim), action], dim=-1)
        output = self.fc(input)
        return output.squeeze(-1)


class DuelingNetWork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, hidden_dim, hidden_layer, ope_dim, gat_hidden_dim, gat_output_dim):
        '''

        :param alpha: Learning Rate
        :param state_dim: OpeGAT output dim
        :param action_dim: OpeFeature dim+InsertGAT output dim
        :param hidden_dim: RLModel hidden size
        :param ope_dim: OpeFeature dim
        :param gat_hidden_dim: GAT hidden size
        :param gat_output_dim: GAT Output dim
        '''
        super(DuelingNetWork, self).__init__()
        self.state_gat = OpeModel(ope_dim, gat_hidden_dim, gat_output_dim)
        # self.job_rnn = nn.LSTM(input_size=gat_output_dim, hidden_size=gat_output_dim, num_layers=1,
        #                        bias=True, batch_first=False, dropout=0.0, bidirectional=False)
        # self.mach_rnn = nn.LSTM(input_size=gat_output_dim, hidden_size=gat_output_dim, num_layers=1,
        #                         bias=True, batch_first=False, dropout=0.0, bidirectional=False)
        self.state_rnn = nn.LSTM(input_size=gat_output_dim, hidden_size=gat_output_dim, num_layers=1,
                                 bias=True, batch_first=False, dropout=0.0, bidirectional=False)
        self.job_gat = InsertPositionModel(ope_dim, gat_hidden_dim, gat_output_dim)
        self.mach_gat = InsertPositionModel(ope_dim, gat_hidden_dim, gat_output_dim)
        self.V_net = VNetwork(alpha, state_dim, action_dim, hidden_dim, hidden_layer)
        self.A_net = ANetwork(alpha, state_dim, action_dim, hidden_dim, hidden_layer)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, node_fea, node_adj, h, c):
        node_gat = self.state_gat(node_fea, node_adj)
        state = node_gat.mean(dim=-2)
        rnn_state, (hn, cn) = self.state_rnn(state.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
        rnn_state = rnn_state.squeeze(0)
        hn = hn.squeeze(0)
        cn = cn.squeeze(0)
        V = self.V_net(rnn_state)
        # node_num = node_fea.shape[1]
        action = node_fea

        A = self.A_net(rnn_state, action)
        return V, A, hn, cn

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class DuelingDQN:
    def __init__(self, alpha, state_dim, action_dim, hidden_dim, hidden_layer,
                 ope_dim, gat_hidden_dim, gat_output_dim,
                 ckpt_dir,
                 ope_num, machine_num,
                 gamma=0.99, tau=0.005, epsilon=0.9, eps_end=0.01, eps_dec=5e-4,
                 max_size=1000000, batch_size=256,
                 summer=None):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.checkpoint_dir = ckpt_dir
        self.action_space = [i for i in range(action_dim)]

        self.q_eval = DuelingNetWork(alpha, state_dim, action_dim,
                                     hidden_dim, hidden_layer,
                                     ope_dim, gat_hidden_dim, gat_output_dim)
        self.q_target = DuelingNetWork(alpha, state_dim, action_dim,
                                       hidden_dim, hidden_layer,
                                       ope_dim, gat_hidden_dim, gat_output_dim)

        self.memory = ReplayBuffer.ReplayBuffer(max_size, batch_size, ope_num, machine_num, gat_output_dim)
        self.update_network_parameters(tau=1.0)
        self.summer_step = 0

        self.summer = summer

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, node_fea, node_adj, action, node_fea_, node_adj_):
        self.memory.step_store(node_fea, node_adj, action, node_fea_, node_adj_)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, state, h, c, isTrain=True):
        action_mask = state[2]
        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(np.argwhere(action_mask[0] != -np.inf).reshape(-1))
            node_fea = T.tensor(state[0], dtype=T.float).to(device)
            node_adj = T.tensor(state[1], dtype=T.int32).to(device)
            h = T.tensor(h, dtype=T.float).to(device)
            c = T.tensor(c, dtype=T.float).to(device)
            _, A, hn, cn = self.q_eval.forward(node_fea, node_adj, h, c)
        else:
            node_fea = T.tensor(state[0], dtype=T.float).to(device)
            node_adj = T.tensor(state[1], dtype=T.int32).to(device)
            action_mask = T.tensor(state[2], dtype=T.float).to(device)
            h = T.tensor(h, dtype=T.float).to(device)
            c = T.tensor(c, dtype=T.float).to(device)
            _, A, hn, cn = self.q_eval.forward(node_fea, node_adj, h, c)
            A = action_mask + A
            action = T.argmax(A).item()
        return action, hn.detach().cpu().numpy(), cn.detach().cpu().numpy()

    def learn(self):

        [node_fea, node_adj, action_mask,
         h, c, action, reward,
         next_node_fea, next_node_adj,
         next_action_mask,
         hn, cn, done] = self.memory.sample()
        batch_idx = T.arange(len(action), dtype=T.long).to(device)
        ope_fea_tensor = T.tensor(np.array(node_fea), dtype=T.float).to(device)
        ope_adj_tensor = T.tensor(np.array(node_adj), dtype=T.float).to(device)

        h_tensor = T.tensor(np.array(h), dtype=T.float).to(device)
        c_tensor = T.tensor(np.array(c), dtype=T.float).to(device)

        actions_tensor = T.tensor(np.array(action), dtype=T.long).to(device)
        rewards_tensor = T.tensor(np.array(reward), dtype=T.float).to(device)

        next_ope_fea_tensor = T.tensor(np.array(next_node_fea), dtype=T.float).to(device)
        next_ope_adj_tensor = T.tensor(np.array(next_node_adj), dtype=T.float).to(device)
        next_action_mask_tensor = T.tensor(np.array(next_action_mask), dtype=T.float).to(device)
        hn_tensor = T.tensor(np.array(hn), dtype=T.float).to(device)
        cn_tensor = T.tensor(np.array(cn), dtype=T.float).to(device)
        done_tensor = T.tensor(np.array(done), dtype=T.float).to(device)

        with T.no_grad():
            V_, A_, _, _ = self.q_target.forward(next_ope_fea_tensor,
                                                 next_ope_adj_tensor,
                                                 hn_tensor,
                                                 cn_tensor)
            q_ = V_ + A_ - T.mean(A_, dim=-1, keepdim=True)
            target = T.zeros(rewards_tensor.shape[0], dtype=T.float).to(device)
            false_index = torch.where(done_tensor == 0)
            true_index = torch.where(done_tensor == 1)
            target[false_index] = rewards_tensor[false_index] + self.gamma * \
                                  T.max(q_[false_index] + next_action_mask_tensor[false_index], dim=-1)[0]
            target[true_index] = rewards_tensor[true_index]
        V, A, _, _ = self.q_eval.forward(ope_fea_tensor,
                                         ope_adj_tensor,
                                         h_tensor,
                                         c_tensor)
        q = (V + A - T.mean(A, dim=-1, keepdim=True))[batch_idx, actions_tensor]

        loss = F.mse_loss(q, target.detach())
        self.summer.add_scalar('Gloss', loss, self.summer_step)
        self.summer_step += 1

        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()

    def save_models(self, model_info):
        self.q_eval.save_checkpoint(
            self.checkpoint_dir + '{}_{}_{}_{}_{}.pth'.format(model_info['seed'], model_info['reward_type'],
                                                              model_info['episode'], model_info['info'],
                                                              model_info['mode_structure']))

        self.q_target.save_checkpoint(
            self.checkpoint_dir + '{}_{}_{}_{}_{}.pth'.format(model_info['seed'], model_info['reward_type'],
                                                              model_info['episode'], model_info['info'],
                                                              model_info['mode_structure']))
        print('Saving network successfully!')

    def load_models(self, model_info):
        if type(model_info) == str:
            self.q_eval.load_checkpoint(model_info)
            self.q_target.load_checkpoint(model_info)
        else:
            self.q_eval.load_checkpoint(
                self.checkpoint_dir + '{}_{}_{}_{}_{}.pth'.format(model_info['seed'], model_info['reward_type'],
                                                                  model_info['episode'], model_info['info'],
                                                                  model_info['model_structure']))
            self.q_target.load_checkpoint(
                self.checkpoint_dir + '{}_{}_{}_{}_{}.pth'.format(model_info['seed'],
                                                                  model_info['reward_type'],
                                                                  model_info['episode'], model_info['info'],
                                                                  model_info['model_structure']))
        print('Loading network successfully!')

#
# if __name__ == '__main__':
#     torch.manual_seed(1)
#     np.random.seed(1)
#     env = FJSPEnviroment()
#     ope_fea, ope_adj, insert_adj, insert_mask, init_cmax = env.reset()
#     ope_fea = np.stack([ope_fea[0], ope_fea[0]])
#     ope_adj = np.stack([ope_adj[0], ope_adj[0]])
#     insert_adj = np.stack([insert_adj[0], insert_adj[0]])
#     insert_mask = np.stack([insert_mask[0], insert_mask[0]])
#     ope_fea = torch.tensor(ope_fea, dtype=torch.float32).to(device)
#     ope_adj = torch.tensor(ope_adj, dtype=torch.int).to(device)
#     insert_adj = torch.tensor(insert_adj, dtype=torch.int).to(device)
#     insert_mask = torch.tensor(insert_mask).to(device)
#     # openet = OpeModel(7, 8, 8, 4)
#     # insnet = InsertPositionModel(16, 8, 8, 4)
#     # policy = Actor(32, 32, 1)
#     DQN = DuelingNetWork(0.9, 8, 15, 64, 7, 8, 8)
#     for i in range(500):
#         V, A = DQN(ope_fea, ope_adj, insert_adj, insert_mask)
#         ope_state_np = ope_state.detach().numpy()
#         insert_state = insnet(ope_state, insert_adj)
#         insert_state_np = insert_state.detach().numpy()
#         a_prob = policy(ope_state, insert_state, insert_mask)
#         dist = Categorical(a_prob)
#         action = dist.sample().detach().numpy()
#         ope_fea, ope_adj, insert_adj, insert_mask, cmax, _, _ = env.step(action[0])
#         print(i, ':', cmax)
