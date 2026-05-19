import numpy as np


class ProblemData:
    def __init__(
        self,
        f,
        N: int,
        a: list[np.float64],
        finish_time: np.float64,
        h: np.float64,
        tao: np.float64,
        viscosity: np.float64,
        X0: np.float64,
        X1: np.float64,
        maxIters = 10000,
        maxLayers = 10000
    ):
        self.N = N
        self.a = a
        self.finish_time = finish_time
        self.h = h
        self.viscosity = viscosity
        self.f = f
        self.cfl = 0.2
        self.tao = tao
        self.X0 = X0
        self.X1 = X1
        self.maxIters = maxIters
        self.maxLayers = maxLayers
        self.EPS: np.float64 = np.float64(1e-18)

