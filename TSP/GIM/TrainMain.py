from GIModel import *
import numpy as np


def main():
    update_model_dict = {1: 'Improve', -1: 'Generate'}
    gim = GIM(config_path="./config.json")
    gim.load()
    ave_cmax = 0
    ep_count = 0
    print_count = 0
    updata_model_flag = 1
    for episode in range(1,gim.config["CommonTrainParas"]["max_episodes"] + 1):
        end_cmax = gim.model[gim.config["CommonTrainParas"]["train_model"]]()
        gim.update(repeat=5, model=update_model_dict[updata_model_flag])
        ave_cmax += end_cmax
        ep_count += 1
        if episode % 500 == 0:
            gim.save(episode)
        if episode % 2000 == 0:  # 变换更新参数的模型
            updata_model_flag *= -1
        if ep_count % len(gim.ie.graph_list) == 0:
            ave_cmax = ave_cmax / len(gim.ie.graph_list)
            gim.summer.add_scalar("EP Cmax ", ave_cmax, print_count)
            print(
                "Episode : {} \t\t EP Cmax : {:6.2f}".format(
                    ep_count,
                    ave_cmax,
                ))
            print_count += 1
            ave_cmax = 0


def mainG():
    with open('./config.json', 'r') as load_f:
        config = json.load(load_f)

    env=GEnviroment.FJSPEnviroment(dir_path='./dataset/' + config['CommonTrainParas']['dataset'])
    time_list = time.ctime().split(' ')
    if '' in time_list:
        time_list.remove('')
    time_clock = time_list[3].split(':')
    time_clock_str = time_clock[0] + '.' + time_clock[1] + '.' + time_clock[2]
    time_list[3] = time_clock_str
    if '' in time_list: time_list.remove('')
    time_str = time_list[1] + '_' + time_list[2] + '_' + time_list[3]
    ope_num = env.graph_list[0].size

    Summer = SummaryWriter('./runs/{}_{}_{}/{}'.format(config['CommonTrainParas']['random_seed'],
                                                       config['CommonTrainParas']['reward_type'],
                                                       config['CommonTrainParas']['save_info'],
                                                       time_str))

    agent=GDuelingDQN.DuelingDQN(alpha=config['GenerateTrainParas']['alpha'],
                                         state_dim=config['GenerateLoadParas']['gat_output_dim'],
                                         action_dim=config['GenerateLoadParas']['gat_output_dim'] * 3,
                                         hidden_dim=config['GenerateLoadParas']['hidden_dim'],
                                         hidden_layer=config['GenerateLoadParas']['hidden_layer_num'],
                                         ope_dim=config['GenerateLoadParas']['ope_dim'],
                                         gat_hidden_dim=config['GenerateLoadParas']['gat_hidden_dim'],
                                         gat_output_dim=config['GenerateLoadParas']['gat_output_dim'],
                                         ckpt_dir=config['GenerateLoadParas']['model_path'],
                                         ope_num=ope_num,
                                         machine_num=0,
                                         gamma=config['GenerateTrainParas']['gamma'],
                                         tau=config['GenerateTrainParas']['tau'],
                                         epsilon=config['GenerateTrainParas']['epsilon'],
                                         eps_end=config['GenerateTrainParas']['eps_end'],
                                         eps_dec=config['GenerateTrainParas']['eps_dec'],
                                         max_size=config['GenerateTrainParas']['max_size'],
                                         batch_size=config['GenerateTrainParas']['batch_size'],
                                         summer=Summer)
    ep_count = 0
    print_count = 0
    for episode in range(config["CommonTrainParas"]["max_episodes"]):
        env.reset()

        if episode % 500 == 0:
            gim.save(episode)
        if episode % 2000 == 0:  # 变换更新参数的模型
            updata_model_flag *= -1
        if ep_count % len(gim.ie.graph_list) == 0:
            ave_cmax = ave_cmax / len(gim.ie.graph_list)
            gim.summer.add_scalar("EP Cmax ", ave_cmax, print_count)
            print(
                "Episode : {} \t\t EP Cmax : {:6.2f}".format(
                    ep_count,
                    ave_cmax,
                ))
            print_count += 1
            ave_cmax = 0
    return 0
if __name__ == '__main__':
    mainG()
