
import numpy as np
import argparse
from utils import plot_learning_curve, create_directory
from DuelingDQN import DuelingDQN
from Enviroment import *
from torch.utils.tensorboard import SummaryWriter
import json
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DuelingDQN/')
parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')
args = parser.parse_args()


def main():
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    train_paras = load_dict['TrainParas']
    load_model_paras = load_dict['LoadModelParas']
    # 加载本次训练参数
    max_episodes = train_paras['max_episodes']
    ep_max_step = train_paras['ep_max_step']
    random_seed = train_paras['random_seed']
    alpha = train_paras['alpha']
    gamma = train_paras['gamma']
    tau = train_paras['tau']
    epsilon = train_paras['epsilon']
    eps_end = train_paras['eps_end']
    eps_dec = train_paras['eps_dec']
    max_size = train_paras['max_size']
    batch_size = train_paras['batch_size']
    reward_type = train_paras['reward_type']
    dataset = train_paras['dataset']
    reset_mode = train_paras['reset_mode']
    save_info = train_paras['save_info']
    save_model_info = {}
    save_model_info['seed'] = random_seed
    save_model_info['reward_type'] = reward_type
    save_model_info['info'] = save_info
    # 加载预训练模型参数
    ope_dim = load_model_paras['ope_dim']
    gat_hidden_dim = load_model_paras['gat_hidden_dim']
    gat_output_dim = load_model_paras['gat_output_dim']
    state_dim = gat_output_dim
    action_dim = gat_output_dim * 3
    hidden_dim = load_model_paras['hidden_dim']
    hidden_layer_num = load_model_paras['hidden_layer_num']
    load_reward_type = load_model_paras['load_reward_type']
    load_seed = load_model_paras['load_seed']
    load_ep = load_model_paras['load_ep']
    load_info = load_model_paras['load_info']
    load_model_info = {}
    load_model_info['seed'] = load_seed
    load_model_info['reward_type'] = load_reward_type
    load_model_info['episode'] = load_ep
    load_model_info['info'] = load_info
    load_model_info['model_structure'] = "{}_{}_{}_{}_{}".format(ope_dim, gat_hidden_dim, gat_output_dim, hidden_dim,
                                                                 hidden_layer_num)
    save_model_info['mode_structure'] = "{}_{}_{}_{}_{}".format(ope_dim, gat_hidden_dim, gat_output_dim, hidden_dim,
                                                                hidden_layer_num)
    time_list = time.ctime().split(' ')
    if '' in time_list:
        time_list.remove('')
    time_clock = time_list[3].split(':')
    time_clock_str = time_clock[0] + '.' + time_clock[1] + '.' + time_clock[2]
    time_list[3] = time_clock_str
    if '' in time_list: time_list.remove('')
    # time_str = ''
    time_str = time_list[1] + '_' + time_list[2] + '_' + time_list[3]
    save_model_info['time'] = time_str

    # random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    env = FJSPEnviroment(dir_path='../dataset/' + dataset)
    env.reset()
    ope_num = env.graph.ope_num
    machine_num = env.graph.machine_num
    job_num = env.graph.job_num
    Summer = SummaryWriter('./runs/{}_{}_{}/{}'.format(save_model_info['seed'],
                                                       save_model_info['reward_type'],
                                                       save_model_info['info'],
                                                       save_model_info['time']))

    agent = DuelingDQN(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                       hidden_dim=hidden_dim, hidden_layer=hidden_layer_num,
                       ope_dim=ope_dim, gat_hidden_dim=gat_hidden_dim,
                       gat_output_dim=gat_output_dim, ckpt_dir=args.ckpt_dir, ope_num=ope_num, machine_num=machine_num,
                       gamma=gamma, tau=tau, epsilon=epsilon,
                       eps_end=eps_end, eps_dec=eps_dec, max_size=max_size, batch_size=batch_size,
                       summer=Summer)
    if load_ep:
        agent.load_models(load_model_info)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    reward_info = {}

    insert_num = env.graph.ope_num + env.graph.machine_num
    ep_ope_fea = np.zeros([ep_max_step, ope_num + 2, ope_dim])
    ep_ope_adj = np.zeros([ep_max_step, ope_num + 2, ope_num + 2])
    ep_job_adj = np.zeros([ep_max_step, job_num, ope_num + 2])
    ep_mach_adj = np.zeros([ep_max_step, machine_num, ope_num + 2])
    ep_action_mask = np.zeros([ep_max_step, job_num * machine_num])
    ep_job_select = np.zeros([ep_max_step, job_num])
    ep_h = np.zeros([ep_max_step, gat_output_dim])
    ep_c = np.zeros([ep_max_step, gat_output_dim])
    ep_action = np.zeros([ep_max_step])
    ep_reward = np.zeros([ep_max_step])
    ep_next_ope_fea = np.zeros([ep_max_step, ope_num + 2, ope_dim])
    ep_next_ope_adj = np.zeros([ep_max_step, ope_num + 2, ope_num + 2])
    ep_next_job_adj = np.zeros([ep_max_step, job_num, ope_num + 2])
    ep_next_mach_adj = np.zeros([ep_max_step, machine_num, ope_num + 2])
    ep_next_action_mask = np.zeros([ep_max_step, job_num * machine_num])
    ep_next_job_select = np.zeros([ep_max_step, job_num])
    ep_hn = np.zeros([ep_max_step, gat_output_dim])
    ep_cn = np.zeros([ep_max_step, gat_output_dim])
    ep_done = np.zeros([ep_max_step])

    total_step = 0
    for episode in range(max_episodes):
        '''
        state:
            np.expand_dims(ope_fea, 0),
            np.expand_dims(ope_adj, 0),
            np.expand_dims(job_adj.T, 0),
            np.expand_dims(mach_adj.T, 0),
            np.expand_dims(action_mask, 0),
            np.expand_dims(pre_cmax, 0),
            np.expand_dims(ope_select, 0)
        '''
        state, done = env.reset(init_mode='Recurrent')
        reward_info['init_cmax'] = state[5]
        cmax_buffer = [state[5]]
        h = np.zeros((1, gat_output_dim))
        c = np.zeros((1, gat_output_dim))
        step = 0
        while not done:
            ep_ope_fea[step] = state[0]
            ep_ope_adj[step] = state[1]
            ep_job_adj[step] = state[2]
            ep_mach_adj[step] = state[3]
            ep_action_mask[step] = state[4]
            ep_job_select[step] = state[6]
            ep_h[step] = h
            ep_c[step] = c
            action, hn, cn = agent.choose_action(state, h, c, isTrain=True)
            n_state, done, _ = env.step(action)
            total_step += 1

            ep_next_ope_fea[step] = n_state[0]
            ep_next_ope_adj[step] = n_state[1]
            ep_next_job_adj[step] = n_state[2]
            ep_next_mach_adj[step] = n_state[3]
            ep_next_action_mask[step] = n_state[4]
            ep_next_job_select[step] = n_state[6]
            ep_hn[step] = hn
            ep_cn[step] = cn
            ep_done[step] = done
            ep_action[step] = action
            cmax_buffer.append(n_state[5])
            state = n_state
            h = hn
            c = cn
            step += 1

        rewards = env.reward(reward_type, cmax_buffer, reward_info)
        ep_buffer_size = len(rewards)
        ep_reward[:ep_buffer_size] = rewards
        agent.memory.store(ep_ope_fea[:ep_buffer_size],
                           ep_ope_adj[:ep_buffer_size],
                           ep_job_adj[:ep_buffer_size],
                           ep_mach_adj[:ep_buffer_size],
                           ep_action_mask[:ep_buffer_size],
                           ep_job_select[:ep_buffer_size],
                           ep_h[:ep_buffer_size],
                           ep_c[:ep_buffer_size],
                           ep_action[:ep_buffer_size],
                           ep_reward[:ep_buffer_size],
                           ep_next_ope_fea[:ep_buffer_size],
                           ep_next_ope_adj[:ep_buffer_size],
                           ep_next_job_adj[:ep_buffer_size],
                           ep_next_mach_adj[:ep_buffer_size],
                           ep_next_action_mask[:ep_buffer_size],
                           ep_next_job_select[:ep_buffer_size],
                           ep_hn[:ep_buffer_size],
                           ep_cn[:ep_buffer_size],
                           ep_done[:ep_buffer_size],
                           ep_buffer_size,
                           )

        best_cmax = cmax_buffer[-1]
        cmax_decady = cmax_buffer[0] - best_cmax
        reward_sum = sum(rewards)
        Summer.add_scalar("Reward", reward_sum, episode)
        Summer.add_scalar("Cmax decady", cmax_decady, episode)
        print(
            "Episode : {} \t\t Episode Reward : {:6.2f} \t\t Cmax decady: {:6.2f} \t\t Cmax : {:6.2f}".format(
                episode,
                reward_sum,
                cmax_decady, best_cmax))
        if agent.memory.total_count > agent.memory.max_size:
            for i in range(5):
                agent.learn()
        if episode % 500 == 0:
            save_model_info['episode'] = episode
            agent.save_models(save_model_info)


if __name__ == '__main__':
    main()
