import random
import numpy


def generate_tsp(dir_path, instance_num=100, instance_size=20, instance_dim=2):
    for num in range(instance_num):
        data = numpy.zeros((instance_size, instance_dim))
        for j in range(instance_size):
            data[j] = numpy.random.uniform(0, 1, instance_dim)
        file_path = '{}/tsp_data_{}_{}_{}_{:0>4d}.npy'.format(dir_path, instance_num, instance_size, instance_dim, num)
        numpy.save(file_path, data)


if __name__ == '__main__':
    dir_path = './dataset/50'
    instance_num = 100
    instance_size = 50
    instance_dim = 2
    generate_tsp(dir_path, instance_num, instance_size, instance_dim)
