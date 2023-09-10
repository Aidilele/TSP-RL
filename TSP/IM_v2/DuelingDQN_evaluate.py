import numpy as np
import argparse
from utils import plot_learning_curve, create_directory
from DuelingDQN import DuelingDQN
from Enviroment import *
import torch
import json
import time



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    render = train_paras['render']
    save_info = train_paras['save_info']
    save_model_info = {}
    save_model_info['seed'] = random_seed
    save_model_info['reward_type'] = reward_type
    save_model_info['info'] = save_info

    # 加载预训练模型参数
    ope_dim = load_model_paras['ope_dim']
    gat_hidden_dim = load_model_paras['gat_hidden_dim']
    gat_output_dim = load_model_paras['gat_output_dim']
    state_dim = gat_output_dim * 2 + ope_dim
    action_dim = gat_output_dim + state_dim
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

    # random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    env = FJSPEnviroment(dir_path='./dataset/' + dataset)
    env.reset()
    ope_num = env.graph.size
    machine_num =0
    agent = DuelingDQN(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                       hidden_dim=hidden_dim, hidden_layer=hidden_layer_num,
                       ope_dim=ope_dim, gat_hidden_dim=gat_hidden_dim,
                       gat_output_dim=gat_output_dim, ckpt_dir='./checkpoints/DuelingDQN/', ope_num=ope_num, machine_num=machine_num,
                       gamma=gamma, tau=tau, epsilon=0.0,
                       eps_end=0.0, eps_dec=0.0, max_size=max_size, batch_size=batch_size)
    if load_ep:
        agent.load_models(load_model_info)
    create_directory('./checkpoints/DuelingDQN/', sub_dirs=['Q_eval', 'Q_target'])
    reward_info = {}
    result = []
    for instance_index in range(len(env.graph_list)):

        state, done = env.eva_reset(instance_index)
        # if render:
        #     env.render()
        # reward_info['init_cmax'] = init_cmax
        cmax_buffer = []
        h = np.zeros((1, state_dim))
        c = np.zeros((1, state_dim))
        step = 0
        for i in range(env.graph.size - env.graph.improve_deepth):
            for j in range(3):
                action, hn, cn = agent.choose_action(state, h, c, isTrain=False)
                n_state, reward, done, _ = env.step(action)
                cmax_buffer.append(n_state[7])
                state = n_state
                h = hn
                c = cn
                step += 1
            env.graph.slider_window()
        # if render:
        #     env.render()

        # print(cmax_buffer)
        best_cmax = round(min(cmax_buffer), 2)
        cmax_decady = 0

        print(
            "Instance : {} \t\t Episode Reward : {} \t\t Cmax decady: {} \t\t Best Cmax : {}".format(
                instance_index,
                round(best_cmax, 1),
                round(cmax_decady, 1), best_cmax))
        result.append(best_cmax)
    print(sum(result)/len(result))
    # for i in result:
    #     print(i)



if __name__ == '__main__':
    main()
