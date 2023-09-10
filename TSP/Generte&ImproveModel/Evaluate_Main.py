from GIModel import *
import numpy as np


def main():
    gim = GIM()
    gim.load()
    for ins_index in range(100):
        state, done = gim.ge.reset(ins_index, solution=0)
        hg = np.zeros((1, gim.config['GenerateLoadParas']['gat_output_dim']))
        cg = np.zeros((1, gim.config['GenerateLoadParas']['gat_output_dim']))
        for i in range(50):
            gaction, hng, cng = gim.ga.choose_action(state, hg, cg, isTrain=True)
            _ = gim.ge.step(gaction)
            solution = gim.ge.graph.solution
            hi = np.zeros((1, gim.config['ImproveLoadParas']['gat_output_dim']))
            ci = np.zeros((1, gim.config['ImproveLoadParas']['gat_output_dim']))
            ope_fea, ope_adj, insert_adj, insert_mask, init_cmax = gim.ie.reset(ins_index, solution)
            solution_buffer = [solution]
            cmax_buffer = [init_cmax]
            for improve_step in range(2):
                iaction, hni, cni = gim.ia.choose_action(ope_fea, ope_adj, insert_adj, insert_mask, hi, ci,
                                                         isTrain=True)
                next_ope_fea, next_ope_adj, next_insert_adj, next_insert_mask, cmax, done, _ = gim.ie.step(iaction)
                if cmax == -1:
                    break
                solution_buffer.append(gim.ie.graph.solution)
                cmax_buffer.append(cmax)
                ope_fea = next_ope_fea
                ope_adj = next_ope_adj
                insert_adj = next_insert_adj
                insert_mask = next_insert_mask
                hi = hni
                ci = cni

            solution = solution_buffer[np.argmin(cmax_buffer)]
            n_state, done = gim.ge.reset(ins_index, solution)
            state = n_state
            hg = hng
            cg = cng
        print(min(cmax_buffer))


if __name__ == '__main__':
    main()
