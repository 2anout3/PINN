from typing import Tuple
import numpy as np
from diff import create_taoLh_op, create_viscosity_op
from problem_data import ProblemData
from shiftoperator import shift_op



def get_solution_constant_dt(init_conds: np.ndarray, pData: ProblemData) -> Tuple:
    finish_time = pData.finish_time
    tao = pData.tao
    Nt = np.min([pData.maxLayers, int(round(finish_time / tao)) + 1])
    solution = np.zeros((Nt,) + init_conds.shape)
    solution[0] = init_conds

    iter = fill_solution(solution, pData)
    return solution, iter


def fill_solution(solution: np.ndarray, pData: ProblemData):
    tao = pData.tao
    a = pData.a
    finish_time = pData.finish_time
    h = pData.h
    viscosity = pData.viscosity
    f = pData.f

    max_layers = np.min([pData.maxLayers, int(round(finish_time / tao)) + 1])
    curr_time: np.float64 = tao
    iter = 1

    W_op = create_viscosity_op(viscosity)
    taoLh_op = create_taoLh_op(a, h, tao)
    buffer = np.zeros_like(solution[0])


    while (finish_time - curr_time > -pData.EPS and iter < pData.maxIters):
        curr_index = iter % max_layers

        prev_layer = solution[(iter - 1) % max_layers]
        curr_layer = solution[curr_index]

        fill_curr_layer(prev_layer, curr_layer, buffer, f, taoLh_op, W_op)

        iter += 1
        curr_time += tao



    print(f"Time: {curr_time}")


    print(f"Final time: {curr_time}")
    print(f"Final iter: {iter}")
    return iter - 1


def fill_curr_layer(prev: np.ndarray, curr: np.ndarray, buffer: np.ndarray,
                    f, taoLh_op: shift_op, W_op: shift_op):
    curr.fill(0)
    # 1
    curr[:] = prev
    buffer[:] = np.apply_along_axis(f, axis=1, arr=prev)
    taoLh_op.apply_and_store(buffer, curr)

    #2
    buffer[:] = 3 * prev + curr
    curr[:] = np.apply_along_axis(f, axis=1, arr=curr)
    taoLh_op.apply_and_store(curr, buffer)
    buffer /= 4

    #3
    curr[:] = prev + 2 * buffer
    buffer[:] = np.apply_along_axis(f, axis=1, arr=buffer)
    (2 * taoLh_op).apply_and_store(buffer, curr)
    curr[:] /= 3

    W_op.apply_and_store(prev, curr)
    curr[0].fill(0)
    curr[-1].fill(0)
