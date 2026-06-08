import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn, optim

from nfd_solver import NFD_Solver
from pinn_solver import PINN_Solver
from problem_data import ProblemData


DT_TO_H_RATIO = np.float64(0.05)
INITIAL_COLLOCATION_COUNT = 100


def initCondFunc(x: np.ndarray):
    eps = np.finfo(np.float64).eps
    res = -np.sin(np.pi * x)
    res[np.abs(res) < eps] = 0
    return res


def createTimeGridForSpace(
    x: np.ndarray, timeStart: np.float64, timeEnd: np.float64
) -> np.ndarray:
    h = x[1] - x[0]
    dt = DT_TO_H_RATIO * h
    time_len = timeEnd - timeStart
    Nt = int(round(time_len / dt))
    return timeStart + dt * np.arange(Nt + 1, dtype=np.float64)


# создание массива граничных условий, первые xLen значений соответсвуют
# значениями в нулевой момент времени
def createBoundaryDataFromArrays(
    T: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xLen = len(X)
    tLen = len(T)
    boundaryLen = xLen + 2 * tLen - 2
    boundaryPoints = np.empty((boundaryLen, 2), dtype=np.float64)

    boundaryPoints[:xLen, 0] = T[0]
    boundaryPoints[:xLen, 1] = X
    boundaryPoints[xLen : xLen + tLen - 1, 0] = T[1:]
    boundaryPoints[xLen : xLen + tLen - 1, 1] = X[0]
    boundaryPoints[xLen + tLen - 1 :, 0] = T[1:]
    boundaryPoints[xLen + tLen - 1 :, 1] = X[-1]

    init = initCondFunc(X[:, None])
    boundaryValues = np.zeros((boundaryLen, init.shape[1]), dtype=np.float64)
    boundaryValues[:xLen, :] = init
    return boundaryPoints, boundaryValues


# выбор случайных точек на сетке
def createCollocationDataFromArrays(
    T: np.ndarray, X: np.ndarray, size: int
) -> torch.Tensor:
    availableCollocationCount = (len(T) - 1) * (len(X) - 2)
    size = min(size, availableCollocationCount)
    idx = torch.randperm(availableCollocationCount)[:size].numpy()
    x_count = len(X) - 2
    t_idx = idx // x_count + 1
    x_idx = idx % x_count + 1

    res = torch.empty(size, 2, dtype=torch.float64)
    res[:, 0] = torch.from_numpy(T[t_idx])
    res[:, 1] = torch.from_numpy(X[x_idx])

    return res


def pde(p, u):
    dudp = torch.autograd.grad(u, p, torch.ones_like(u), create_graph=True)[0]

    dudt = dudp[:, 0:1]
    dudx = dudp[:, 1:2]

    return dudt + u * dudx


def createModel(
    output_size,
    layers_num=10,
    width=64,
):
    layers = [nn.Linear(2, width), nn.Tanh()]
    for _ in range(layers_num - 1):
        layers.extend((nn.Linear(width, width), nn.Tanh()))
    layers.append(nn.Linear(width, output_size))
    model = nn.Sequential(*layers).double()
    return model


def f(x):
    res = np.zeros_like(x)
    try:
        with np.errstate(over="raise", invalid="raise"):
            res = np.pow(x, 2) / 2
    except FloatingPointError as exc:
        print(res)
        raise Exception("Failed to calculate flux function") from exc
    return res


def createNFDSolver(Nx, ft, h, dt, X0, X1):
    layer_count = int(round(ft / dt)) + 1
    pd = ProblemData(
        f,
        Nx,
        NFD_Solver._a,
        ft,
        h,
        dt,
        NFD_Solver._viscosity,
        X0,
        X1,
        maxIters=layer_count + 1,
        maxLayers=layer_count,
    )
    return NFD_Solver(pd)


def createQuasiSolution(quasi_space: int):
    xStart = np.float64(-1)
    xEnd = np.float64(1)
    X = np.linspace(xStart, xEnd, quasi_space + 1, dtype=np.float64)
    T = createTimeGridForSpace(X, np.float64(0), np.float64(1))
    _, boundaryValues = createBoundaryDataFromArrays(T, X)

    h = X[1] - X[0]
    dt = T[1] - T[0]
    print(f"quasi: h = {h}, dt = {dt}")
    nfd_solver = createNFDSolver(quasi_space, 1, h, dt, xStart, xEnd)
    inCond = boundaryValues[: quasi_space + 1]
    return T, X, nfd_solver.getSolution(inCond)

def getRandomInitialAndBoundary(
    boundaryPoints: np.ndarray,
    boundaryValues: np.ndarray,
    initial_count: int | None,
    boundary_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_mask = boundaryPoints[:, 0] == 0
    initial_points = boundaryPoints[initial_mask]
    initial_values = boundaryValues[initial_mask]
    side_points = boundaryPoints[~initial_mask]
    side_values = boundaryValues[~initial_mask]

    if initial_count is None:
        initial_count = len(initial_points)
    initial_count = min(initial_count, len(initial_points))
    boundary_count = min(boundary_count, len(side_points))

    initial_idxs = torch.randperm(len(initial_points))[:initial_count]
    boundary_idxs = torch.randperm(len(side_points))[:boundary_count]

    ip = initial_points[initial_idxs]
    iv = initial_values[initial_idxs]
    bp = side_points[boundary_idxs]
    bv = side_values[boundary_idxs]
    return torch.from_numpy(ip), torch.from_numpy(iv), torch.from_numpy(bp), torch.from_numpy(bv)


def createInitialCollocationData(
    boundaryPoints: np.ndarray, count: int = INITIAL_COLLOCATION_COUNT
) -> torch.Tensor:
    initial_mask = boundaryPoints[:, 0] == 0
    initial_points = boundaryPoints[initial_mask]
    count = min(count, len(initial_points))
    idxs = torch.randperm(len(initial_points))[:count].numpy()
    return torch.from_numpy(initial_points[idxs]).double()


def createPinnSolver(
    T: np.ndarray,
    X: np.ndarray,
    boundaryPoints: np.ndarray,
    boundaryValues: np.ndarray,
    layers_num: int,
    layers_width: int,
    initial_count: int | None,
    boundary_count: int,
    collocation_count: int | None,
    initial_collocation_points: torch.Tensor,
) -> PINN_Solver:
    availableCollocationCount = (len(T) - 1) * (len(X) - 2)
    collocationCount = availableCollocationCount // 2
    if collocation_count is not None:
        collocationCount = min(collocation_count, availableCollocationCount)
    print(f"collocationCount = {collocationCount}")
    ip, iv, bp, bv = getRandomInitialAndBoundary(
        boundaryPoints,
        boundaryValues,
        initial_count,
        boundary_count,
    )
    cp = createCollocationDataFromArrays(T, X, collocationCount)
    cp = torch.vstack((cp, initial_collocation_points))

    model = createModel(bv.shape[-1], layers_num, layers_width)
    initializer = nn.init.xavier_uniform_

    optimizer = optim.Adam(model.parameters())
    return PINN_Solver(
        initialPoints=ip,
        initialValues=iv,
        boundaryPoints=bp,
        boundaryValues=bv,
        collocationPoints=cp,
        pdeFn=pde,
        lossFn=nn.MSELoss(reduction="mean"),
        model=model,
        initializer=initializer,
        optimizer=optimizer,
    )


def write_dataset(group: h5py.Group, name: str, data):
    if name in group:
        del group[name]
    return group.create_dataset(name, data=data)


# сохранение модели в hdf5 файл
def write_model(
    group: h5py.Group, solver: PINN_Solver, layers_num: int, layers_width: int
):
    group.attrs["layers_num"] = layers_num
    group.attrs["layers_width"] = layers_width
    group.attrs["output_size"] = solver.iv.shape[-1]

    write_dataset(group, "training_initial_points", solver.ip.detach().cpu().numpy())
    write_dataset(group, "training_boundary_points", solver.bp.detach().cpu().numpy())
    write_dataset(
        group,
        "training_collocation_points",
        solver.cp.detach().cpu().numpy(),
    )

    state_group = group.create_group("state_dict")
    keys = []
    for key, value in solver.model.state_dict().items():
        keys.append(key)
        state_group.create_dataset(key, data=value.detach().cpu().numpy())
    group.attrs["state_keys"] = json.dumps(keys)


# основная функция файла
def generate_hdf5(args):
    torch.manual_seed(args.seed)

    steps = args.steps
    n_start = args.n_start
    x_start = np.float64(-1)
    x_end = np.float64(1)
    Nx = tuple(n_start * 2**i for i in range(steps))
    timeStart = np.float64(0)
    timeEnd = np.float64(1)
    XS = tuple(np.linspace(x_start, x_end, nx + 1, dtype=np.float64) for nx in Nx)
    TS = tuple(createTimeGridForSpace(x, timeStart, timeEnd) for x in XS)

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data_output, "w") as h5:
        # мета данные
        h5.attrs["format"] = "burgers-pinn-nfd"
        h5.attrs["steps"] = steps
        h5.attrs["n_start"] = n_start
        h5.attrs["dt_to_h_ratio"] = DT_TO_H_RATIO
        h5.attrs["time_steps"] = json.dumps([len(t) - 1 for t in TS])
        h5.attrs["time_start"] = timeStart
        h5.attrs["time_end"] = timeEnd

        # полное сохранение квази решения
        quasi_group = h5.create_group("quasi")
        quasi_t, quasi_x, quasi_solution = createQuasiSolution(args.quasi_space)
        write_dataset(quasi_group, "T", quasi_t)
        write_dataset(quasi_group, "X", quasi_x)
        write_dataset(quasi_group, "solution", quasi_solution)

        iterations_group = h5.create_group("iterations")
        for iteration, nx in enumerate(Nx):
            T = TS[iteration]
            X = XS[iteration]
            boundaryPoints, boundaryValues = createBoundaryDataFromArrays(T, X)
            h = X[1] - X[0]
            dt = T[1] - T[0]
            print(f"iteration = {iteration}, Nx = {nx}")
            print(f"h = {h}, dt = {dt}")
            iteration_group = iterations_group.create_group(str(iteration))
            iteration_group.attrs["Nx"] = nx
            iteration_group.attrs["time_steps"] = len(T) - 1
            write_dataset(iteration_group, "T", T)
            write_dataset(iteration_group, "X", X)
            initial_collocation_points = createInitialCollocationData(boundaryPoints)
            write_dataset(
                iteration_group,
                "initial_collocation_points",
                initial_collocation_points.detach().cpu().numpy(),
            )

            solver = createPinnSolver(
                T,
                X,
                boundaryPoints,
                boundaryValues,
                args.model_layers_num,
                args.model_layers_width,
                args.initial_count,
                args.boundary_count,
                args.collocation_count,
                initial_collocation_points,
            )

            if args.adam_epochs > 0:
                solver.trainModel(args.adam_epochs)
            if args.lbfgs_iters > 0:
                solver.trainLBFGS(args.lbfgs_iters)

            inCond = boundaryValues[: nx + 1]
            nfd_solver = createNFDSolver(nx, timeEnd, h, dt, x_start, x_end)
            nfd_solution = nfd_solver.getSolution(inCond)

            # сохранение nfd
            nfd_group = iteration_group.create_group("nfd")
            write_dataset(nfd_group, "solution", nfd_solution)

            # сохранение pinn
            model_group = iteration_group.create_group("model")
            write_model(
                model_group, solver, args.model_layers_num, args.model_layers_width
            )

    print(f"Data saved to {args.data_output}")


def parseInitialCount(value: str) -> int | None:
    if value.lower() == "all":
        return None
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("--initial-count must be non-negative or all")
    return count


def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description="Create and save data for Burgers PINN/NFD visualization."
    )
    parser.add_argument("--quasi-space", type=int, default=100)
    parser.add_argument("--model-layers-num", type=int, default=10)
    parser.add_argument("--model-layers-width", type=int, default=64)
    parser.add_argument(
        "--data-output", type=Path, default=Path("computationData.hdf5")
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--n-start", type=int, default=200)
    parser.add_argument(
        "--initial-count",
        type=parseInitialCount,
        default=None,
        help="Initial-condition points count, or 'all'. Uses all initial points by default.",
    )
    parser.add_argument("--boundary-count", type=int, default=600)
    parser.add_argument(
        "--collocation-count",
        type=int,
        default=None,
        help="Count of collocation points. Uses all inner grid points by default.",
    )
    parser.add_argument("--adam-epochs", type=int, default=10)
    parser.add_argument("--lbfgs-iters", type=int, default=20)
    parser.add_argument(
        "--alternative-collocation-count",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--alternative-adam-epochs", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--alternative-lbfgs-iters",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    generate_hdf5(parseArgs(argv))


if __name__ == "__main__":
    main()
