import numpy as np
from shiftoperator import shift_op

def create_taoLh_op(a, h, tao) -> shift_op:
    result = shift_op(0, 0)
    for i in range(len(a)):
        shift = i + 1
        result += shift_op(shift, a[i]) + shift_op(-shift, -a[i])
    return (tao / h) * result

def create_lambda_op() -> shift_op:
    return shift_op(3) + shift_op(-3) + (-6) * (shift_op(2) + shift_op(-2)) + 15 * (shift_op(1) + shift_op(-1)) + ((-20) * shift_op(0))

def create_viscosity_op(viscosity) -> shift_op:
    return viscosity * create_lambda_op()
