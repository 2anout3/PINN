import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_data import createModel


def load_model(model_group: h5py.Group):
    layers_num = int(model_group.attrs["layers_num"])
    layers_width = int(model_group.attrs["layers_width"])
    output_size = int(model_group.attrs["output_size"])
    model = createModel(
        output_size,
        layers_num,
        layers_width,
    )

    state_keys = json.loads(model_group.attrs["state_keys"])
    state_dict = {}
    for key in state_keys:
        state_dict[key] = torch.from_numpy(model_group["state_dict"][key][()])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def nearest_time_index(T, moment):
    dt = T[1] - T[0]
    time_idx = int(round(moment / dt))
    return min(max(time_idx, 0), len(T) - 1)


def get_pinn_solution(model, T, X, time_idx):
    x_line = torch.from_numpy(X).double()
    t_line = torch.full_like(x_line, fill_value=T[time_idx])
    pinn_points = torch.stack((t_line, x_line), dim=1)

    with torch.no_grad():
        return model(pinn_points).detach().cpu().numpy()[:, 0]


def get_pinn_grid_solution(model, T, X):
    t_grid, x_grid = np.meshgrid(T, X, indexing="ij")
    points = torch.from_numpy(
        np.column_stack((t_grid.ravel(), x_grid.ravel()))
    ).double()

    with torch.no_grad():
        values = model(points).detach().cpu().numpy()[:, 0]
    return values.reshape(len(T), len(X))


def interpolate_quasi_solution(T, X, quasi_T, quasi_X, quasi_solution):
    quasi_on_grid = np.empty((len(T), len(X)), dtype=np.float64)
    for time_idx, time in enumerate(T):
        quasi_time_idx = nearest_time_index(quasi_T, time)
        quasi_at_time = quasi_solution[quasi_time_idx, :, 0]
        quasi_on_grid[time_idx] = np.interp(X, quasi_X, quasi_at_time)
    return quasi_on_grid


def relative_l2_error(solution, reference):
    denominator = np.linalg.norm(reference.ravel(), 2)
    if denominator == 0:
        return np.linalg.norm((solution - reference).ravel(), 2)
    return np.linalg.norm((solution - reference).ravel(), 2) / denominator


def print_relative_l2_errors(
    iteration, nx, T, X, nfd_solution, models, quasi_T, quasi_X, quasi_solution
):
    quasi_on_grid = interpolate_quasi_solution(T, X, quasi_T, quasi_X, quasi_solution)
    print(f"Iteration {iteration}, Nx = {nx}:")
    print(f"  NFD: {relative_l2_error(nfd_solution[:, :, 0], quasi_on_grid):.6e}")
    for name, model in models.items():
        pinn_solution = get_pinn_grid_solution(model, T, X)
        print(f"  {name}: {relative_l2_error(pinn_solution, quasi_on_grid):.6e}")


def get_iteration_values(
    T, X, nfd_solution, models, moment, quasi_T, quasi_X, quasi_solution
):
    time_idx = nearest_time_index(T, moment)
    x_line_np = X
    pinn_solutions = {
        name: get_pinn_solution(model, T, X, time_idx) for name, model in models.items()
    }

    quasi_time_idx = nearest_time_index(quasi_T, T[time_idx])
    quasi_at_time = quasi_solution[quasi_time_idx, :, 0]
    quasi_on_x = np.interp(x_line_np, quasi_X, quasi_at_time)

    return (
        T[time_idx],
        x_line_np,
        nfd_solution[time_idx, :, 0],
        pinn_solutions,
        quasi_on_x,
    )


# отображение решений
def plot_iteration(
    ax, T, X, nfd_solution, models, moment, quasi_T, quasi_X, quasi_solution
):
    time, x_line_np, nfd_at_time, pinn_solutions, _ = get_iteration_values(
        T, X, nfd_solution, models, moment, quasi_T, quasi_X, quasi_solution
    )

    ax.plot(x_line_np, nfd_at_time, label="NFD")
    styles = {"PINN": "--", "PINN Alternative": "-."}
    for name, solution in pinn_solutions.items():
        ax.plot(x_line_np, solution, styles.get(name, "--"), label=name)
    quasi_time_idx = nearest_time_index(quasi_T, time)
    ax.plot(quasi_X, quasi_solution[quasi_time_idx, :, 0], ":", label="Quasi")
    ax.set_title(f"t = {time:g}")
    ax.grid(True)


def plot_error_iteration(
    ax, T, X, nfd_solution, models, moment, quasi_T, quasi_X, quasi_solution
):
    time, x_line_np, nfd_at_time, pinn_solutions, quasi_on_x = get_iteration_values(
        T, X, nfd_solution, models, moment, quasi_T, quasi_X, quasi_solution
    )

    ax.plot(x_line_np, np.abs(nfd_at_time - quasi_on_x), label="|NFD - Quasi|")
    styles = {"PINN": "--", "PINN Alternative": "-."}
    for name, solution in pinn_solutions.items():
        ax.plot(
            x_line_np,
            np.abs(solution - quasi_on_x),
            styles.get(name, "--"),
            label=f"|{name} - Quasi|",
        )
    ax.set_title(f"t = {time:g}")
    ax.grid(True)


def plot_training_points(ax, T, X, model_group, title):
    t_min = T.min()
    t_max = T.max()
    x_min = X.min()
    x_max = X.max()
    rectangle_t = [t_min, t_max, t_max, t_min, t_min]
    rectangle_x = [x_min, x_min, x_max, x_max, x_min]

    ax.plot(rectangle_t, rectangle_x, color="black", linewidth=1)
    if "training_initial_points" in model_group:
        initial_points = model_group["training_initial_points"][()]
        ax.plot(
            initial_points[:, 0],
            initial_points[:, 1],
            "x",
            markersize=4,
            label="initial",
        )
    if "training_boundary_points" in model_group:
        boundary_points = model_group["training_boundary_points"][()]
        ax.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "x",
            markersize=4,
            label="boundary",
        )

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
    ax.grid(True)


def plot_training_points_hdf5(h5, iterations, args):
    model_names = (("model", "PINN"), ("alternative_model", "PINN Alternative"))
    fig, axes = plt.subplots(
        len(iterations),
        len(model_names),
        figsize=(4 * len(model_names), 3 * len(iterations)),
        squeeze=False,
    )

    for row, iteration in enumerate(iterations):
        iteration_group = h5["iterations"][iteration]
        T = iteration_group["T"][()]
        X = iteration_group["X"][()]
        nx = int(iteration_group.attrs["Nx"])
        for col, (group_name, label) in enumerate(model_names):
            ax = axes[row][col]
            if group_name in iteration_group:
                plot_training_points(
                    ax,
                    T,
                    X,
                    iteration_group[group_name],
                    f"Nx = {nx}, {label}",
                )
            else:
                ax.set_axis_off()

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()

    if args.save_plot is not None:
        training_plot = args.save_plot.with_name(
            f"{args.save_plot.stem}_training_points{args.save_plot.suffix}"
        )
        fig.savefig(training_plot, dpi=200)
        print(f"Training points plot saved to {training_plot}")
    return fig


# главная функция файла
def visualize_hdf5(args):
    with h5py.File(args.data_input, "r") as h5:
        quasi_T = h5["quasi"]["T"][()]
        quasi_X = h5["quasi"]["X"][()]
        quasi_solution = h5["quasi"]["solution"][()]
        iterations = sorted(h5["iterations"].keys(), key=int)
        if args.iteration is not None:
            iterations = [str(args.iteration)]

        training_points_fig = None
        if args.plot_training_points:
            training_points_fig = plot_training_points_hdf5(h5, iterations, args)

        moments = args.time_moments
        fig, axes = plt.subplots(
            len(iterations),
            len(moments),
            figsize=(4 * len(moments), 3 * len(iterations)),
            sharex=False,
            sharey=True,
            squeeze=False,
        )
        error_fig, error_axes = plt.subplots(
            len(iterations),
            len(moments),
            figsize=(4 * len(moments), 3 * len(iterations)),
            sharex=False,
            sharey=True,
            squeeze=False,
        )

        for row, iteration in enumerate(iterations):
            iteration_group = h5["iterations"][iteration]
            T = iteration_group["T"][()]
            X = iteration_group["X"][()]
            nfd_solution = iteration_group["nfd"]["solution"][()]
            models = {
                "PINN": load_model(iteration_group["model"]),
                "PINN Alternative": load_model(iteration_group["alternative_model"]),
            }
            print_relative_l2_errors(
                iteration,
                int(iteration_group.attrs["Nx"]),
                T,
                X,
                nfd_solution,
                models,
                quasi_T,
                quasi_X,
                quasi_solution,
            )

            for col, moment in enumerate(moments):
                ax = axes[row][col]
                plot_iteration(
                    ax,
                    T,
                    X,
                    nfd_solution,
                    models,
                    moment,
                    quasi_T,
                    quasi_X,
                    quasi_solution,
                )
                plot_error_iteration(
                    error_axes[row][col],
                    T,
                    X,
                    nfd_solution,
                    models,
                    moment,
                    quasi_T,
                    quasi_X,
                    quasi_solution,
                )
                if col == 0:
                    ax.set_ylabel(f"Nx = {int(iteration_group.attrs['Nx'])}\nu(t, x)")
                    error_axes[row][col].set_ylabel(
                        f"Nx = {int(iteration_group.attrs['Nx'])}\nabsolute error"
                    )
                if row == len(iterations) - 1:
                    ax.set_xlabel("x")
                    error_axes[row][col].set_xlabel("x")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        error_handles, error_labels = error_axes[0][0].get_legend_handles_labels()
        error_fig.legend(error_handles, error_labels, loc="upper right")
        error_fig.tight_layout()
        if args.save_plot is not None:
            args.save_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_plot, dpi=200)
            print(f"Plot saved to {args.save_plot}")
            error_plot = args.save_plot.with_name(
                f"{args.save_plot.stem}_errors{args.save_plot.suffix}"
            )
            error_fig.savefig(error_plot, dpi=200)
            print(f"Error plot saved to {error_plot}")
        if not args.no_plot:
            plt.show()
        elif training_points_fig is not None:
            plt.close(training_points_fig)


def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize Burgers PINN/NFD data saved to HDF5."
    )
    parser.add_argument("--data-input", type=Path, default=Path("computationData.hdf5"))
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument(
        "--time-moments", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75]
    )
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-training-points", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    visualize_hdf5(parseArgs(argv))


if __name__ == "__main__":
    main()
