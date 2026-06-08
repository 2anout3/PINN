import numpy as np
import h5py

from problem_data import ProblemData

class config:
    def __init__(self, data: ProblemData, path = None, ds_solution_name: str = "solution_def", ds_error_name: str = "erorr_def", read: bool = True, max_iters = 1000, max_layers = 1000):
        self.data = data
        self.path = path
        self.use_file_to_store = not path is None
        self.ds_solution_name = ds_solution_name
        self.is_opened = False
        self.file: h5py.File = None
        self.wgroups: dict = {}
        self.read = read

        self.max_iters = max_iters
        self.max_layers = max_layers

    def get_error_group(self):
        return self.wgroups[self.ERROR_GROUP]

    def get_solution_group(self) -> h5py.Group:
        return self.wgroups[self.SOLUTION_GROUP]

    def clear_file(self):
        if self.use_file_to_store:
            self.file.clear()

    def init_files(self):
        self.file = h5py.File(self.path, "a")

    def init_groups(self):
        if (not self.SOLUTION_GROUP in self.file):
            self.file.create_group(self.SOLUTION_GROUP)

        if (not self.ERROR_GROUP in self.file):
            self.file.create_group(self.ERROR_GROUP)

        self.wgroups[self.SOLUTION_GROUP] = self.file[self.SOLUTION_GROUP]
        self.wgroups[self.ERROR_GROUP] = self.file[self.ERROR_GROUP]


    def get_error_runge_p(self) -> h5py.Group:
        return self.wgroups[self.RUNGE_P]

    def get_error_runge_c(self):
        return self.wgroups[self.RUNGE_C]

    def close_files(self):
        self.file.close()

    EPS = np.float64(1e-9)
    SOLUTION_GROUP = "solution"
    ERROR_GROUP = "error"
    SPACE_ERROR_NAME = "space"
    TIME_ERROR_NAME = "time"
    ITERATION_COUNT = "iteration_count"
    SPACE_GRID_SHIFT = "space_shift"
    SPACE_GRID_SIZE = "space_grid_size"
    DIM_AMOUNT = "dim_amount"
    SPACE_ERROR_GROUP = "space"
    ERROR_DATASET_ORDER = "error_order"
    ERROR_DATASET_DELTA = "error_delta"
    ERROR_DATASET_RUNGE_C_TIME = "runge_c_time"
    ERROR_DATASET_RUNGE_P_TIME = "runge_p_time"
    TIME_SHIFT = "time_shift"
    TIMESTAMPS_NAME = "timestamps"
    RUNGE_P = "runge_p"
    RUNGE_C = "runge_c"
    MIN_TIME_SHIFT = 0.00001
    SPACE_START = "space_start"
    SPACE_END = "space_end"
    TIME_START = "time_start"
    TIME_END = "time_end"



