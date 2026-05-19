import numpy as np
from problem_data import ProblemData
from constant import get_solution_constant_dt


class NFD_Solver:
    _a = [
        np.float64(-3) / np.float64(4),
        np.float64(3) / np.float64(20),
        np.float64(-1) / np.float64(60),
    ]

    _viscosity = 1 / np.float64(64)

    def __init__(self, problemData: ProblemData):
        self.pd = problemData

    def loadProblemData(self, problemData: ProblemData):
        self.pd = problemData

    def getSolution(self, initConditions):
       return get_solution_constant_dt(initConditions, self.pd)[0]

