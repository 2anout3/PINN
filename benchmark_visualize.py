import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from generate_data import createBoundaryDataFromArrays, createNFDSolver
from visualize_data import (
    get_pinn_solution,
    get_pinn_grid_solution,
    interpolate_quasi_solution,
    load_model,
    nearest_time_index,
    relative_l2_error,
)


SUMMARY_FIELDS = (
    "nx",
    "stage",
    "case",
    "path",
    "relative_l2_error",
    "adam_epochs",
    "lbfgs_iters",
    "model_layers_width",
    "model_layers_num",
    "initial_count",
    "boundary_count",
    "collocation_count",
)

PARAMETER_LABELS = {
    "adam_epochs": "Количество эпох Adam",
    "lbfgs_iters": "Количество итераций L-BFGS",
    "model_layers_width": "Количество нейронов в скрытом слое",
    "model_layers_num": "Количество скрытых слоев",
    "initial_count": "Количество точек начального условия",
    "boundary_count": "Количество граничных точек",
    "collocation_count": "Количество внутренних точек проверки уравнения",
}

STAGE_TITLES = {
    "width": "Зависимость ошибки от ширины скрытого слоя",
    "layers": "Зависимость ошибки от количества скрытых слоев",
    "initial_count": "Зависимость ошибки от количества точек начального условия",
    "boundary_count": "Зависимость ошибки от количества граничных точек",
    "collocation_count": "Зависимость ошибки от количества внутренних точек",
}


def nx_legend_label(nx: int) -> str:
    return f"N_x = {nx}"


def param_label(param: str) -> str:
    return PARAMETER_LABELS.get(param, param)


def format_initial_count(value) -> str:
    return "все" if str(value) == "all" else str(value)


def load_manifest(input_dir: Path) -> dict:
    with (input_dir / "manifest.json").open(encoding="utf-8") as file:
        return json.load(file)


def read_summary(input_dir: Path, manifest: dict) -> list[dict]:
    summary_path = input_dir / manifest.get("summary_path", "benchmark_summary.csv")
    if not summary_path.exists():
        return []

    rows = []
    with summary_path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row["nx"] = int(row["nx"])
            row["relative_l2_error"] = float(row["relative_l2_error"])
            for key in (
                "adam_epochs",
                "lbfgs_iters",
                "model_layers_width",
                "model_layers_num",
                "boundary_count",
                "collocation_count",
            ):
                if row.get(key) not in ("", None):
                    row[key] = int(row[key])
            rows.append(row)
    return rows


def load_quasi(input_dir: Path, manifest: dict):
    quasi_path = input_dir / manifest.get("quasi_path", "quasi.hdf5")
    with h5py.File(quasi_path, "r") as h5:
        return h5["T"][()], h5["X"][()], h5["solution"][()]


def calculate_case_error(input_dir: Path, case: dict, quasi_data) -> dict:
    h5_path = input_dir / case["path"]
    quasi_T, quasi_X, quasi_solution = quasi_data
    with h5py.File(h5_path, "r") as h5:
        T = h5["T"][()]
        X = h5["X"][()]
        model = load_model(h5["model"])
        pinn_solution = get_pinn_grid_solution(model, T, X)
        quasi_on_grid = interpolate_quasi_solution(
            T, X, quasi_T, quasi_X, quasi_solution
        )
        error = relative_l2_error(pinn_solution, quasi_on_grid)
        row = {
            "nx": int(h5.attrs["Nx"]),
            "stage": h5.attrs["stage"],
            "case": h5.attrs["name"],
            "path": case["path"],
            "relative_l2_error": float(error),
        }
        for attr in SUMMARY_FIELDS:
            if attr not in row and attr in h5.attrs:
                row[attr] = h5.attrs[attr]
        return row


def calculate_summary(input_dir: Path, manifest: dict) -> list[dict]:
    quasi_data = load_quasi(input_dir, manifest)
    rows = []
    for case in manifest["cases"]:
        h5_path = input_dir / case["path"]
        if not h5_path.exists():
            print(f"Skipping missing {h5_path}")
            continue
        print(f"Reading {h5_path}")
        row = calculate_case_error(input_dir, case, quasi_data)
        rows.append(row)
    return rows


def write_summary(rows: list[dict], output: Path):
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rows_for(rows: list[dict], stage: str, nx: int | None = None) -> list[dict]:
    selected = [row for row in rows if row["stage"] == stage]
    if nx is not None:
        selected = [row for row in selected if row["nx"] == nx]
    return selected


def unique_nx(rows: list[dict]) -> list[int]:
    return sorted({int(row["nx"]) for row in rows})


def clipped_log10_error(values):
    return np.log10(np.clip(values, np.finfo(np.float64).tiny, None))


def plot_epochs(rows: list[dict], output_dir: Path):
    epoch_rows = rows_for(rows, "epochs")
    if not epoch_rows:
        return

    all_errors = np.array(
        [row["relative_l2_error"] for row in epoch_rows],
        dtype=np.float64,
    )
    log_errors = clipped_log10_error(all_errors)
    color_min = float(np.nanmin(log_errors))
    color_max = float(np.nanmax(log_errors))

    for nx in unique_nx(epoch_rows):
        stage_rows = rows_for(rows, "epochs", nx)
        if not stage_rows:
            continue

        adam_values = sorted({int(row["adam_epochs"]) for row in stage_rows})
        lbfgs_values = sorted({int(row["lbfgs_iters"]) for row in stage_rows})
        values = np.full((len(adam_values), len(lbfgs_values)), np.nan)
        adam_index = {value: idx for idx, value in enumerate(adam_values)}
        lbfgs_index = {value: idx for idx, value in enumerate(lbfgs_values)}

        for row in stage_rows:
            values[
                adam_index[int(row["adam_epochs"])],
                lbfgs_index[int(row["lbfgs_iters"])],
            ] = row["relative_l2_error"]

        fig, ax = plt.subplots(figsize=(7, 5))
        log_values = clipped_log10_error(values)
        image = ax.imshow(
            log_values,
            origin="lower",
            aspect="auto",
            vmin=color_min,
            vmax=color_max,
        )
        ax.set_xticks(range(len(lbfgs_values)), lbfgs_values)
        ax.set_yticks(range(len(adam_values)), adam_values)
        ax.set_xlabel("Количество итераций L-BFGS")
        ax.set_ylabel("Количество эпох Adam")
        ax.set_title(
            f"N_x = {nx}: десятичный логарифм относительной ошибки L2"
        )
        fig.colorbar(
            image,
            ax=ax,
            label="log10 относительной ошибки L2",
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"epochs_nx_{nx}.png", dpi=200)
        plt.close(fig)


def plot_numeric_stage(rows: list[dict], stage: str, param: str, output_dir: Path):
    stage_rows = rows_for(rows, stage)
    if not stage_rows:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for nx in unique_nx(stage_rows):
        nx_rows = rows_for(rows, stage, nx)
        nx_rows.sort(key=lambda row: int(row[param]))
        x = [int(row[param]) for row in nx_rows]
        y = [row["relative_l2_error"] for row in nx_rows]
        ax.plot(x, y, marker="o", label=nx_legend_label(nx))

    ax.set_xlabel(param_label(param))
    ax.set_ylabel("Относительная ошибка L2")
    ax.set_title(STAGE_TITLES.get(stage, f"Этап подбора параметра: {stage}"))
    # ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{stage}.png", dpi=200)
    plt.close(fig)


def plot_initial_count(rows: list[dict], output_dir: Path):
    stage_rows = rows_for(rows, "initial_count")
    if not stage_rows:
        return

    order = {"10": 0, "50": 1, "100": 2, "all": 3}
    fig, ax = plt.subplots(figsize=(7, 4))
    for nx in unique_nx(stage_rows):
        nx_rows = rows_for(rows, "initial_count", nx)
        nx_rows.sort(key=lambda row: order.get(str(row["initial_count"]), 100))
        labels = [format_initial_count(row["initial_count"]) for row in nx_rows]
        y = [row["relative_l2_error"] for row in nx_rows]
        ax.plot(range(len(labels)), y, marker="o", label=nx_legend_label(nx))

    ax.set_xticks(range(len(order)), [format_initial_count(value) for value in order])
    ax.set_xlabel("Количество точек начального условия")
    ax.set_ylabel("Относительная ошибка L2")
    ax.set_title(STAGE_TITLES["initial_count"])
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "initial_count.png", dpi=200)
    plt.close(fig)


def best_rows_by_nx(rows: list[dict]) -> dict[int, dict]:
    best = {}
    for row in rows:
        nx = int(row["nx"])
        if nx not in best or row["relative_l2_error"] < best[nx]["relative_l2_error"]:
            best[nx] = row
    return best


def create_nfd_solution(T: np.ndarray, X: np.ndarray):
    nx = len(X) - 1
    _, boundary_values = createBoundaryDataFromArrays(T, X)
    h = X[1] - X[0]
    dt = T[1] - T[0]
    nfd_solver = createNFDSolver(nx, T[-1], h, dt, X[0], X[-1])
    return nfd_solver.getSolution(boundary_values[: nx + 1])


def plot_best_pinn_nfd_quasi(
    input_dir: Path,
    rows: list[dict],
    quasi_data,
    time_moments: list[float],
    output_dir: Path,
):
    quasi_T, quasi_X, quasi_solution = quasi_data

    for nx, row in best_rows_by_nx(rows).items():
        h5_path = input_dir / row["path"]
        if not h5_path.exists():
            print(f"Skipping missing {h5_path}")
            continue

        with h5py.File(h5_path, "r") as h5:
            T = h5["T"][()]
            X = h5["X"][()]
            model = load_model(h5["model"])

        nfd_solution = create_nfd_solution(T, X)
        fig, axes = plt.subplots(
            1,
            len(time_moments),
            figsize=(4 * len(time_moments), 3),
            sharey=True,
            squeeze=False,
        )

        for col, moment in enumerate(time_moments):
            ax = axes[0][col]
            time_idx = nearest_time_index(T, moment)
            quasi_time_idx = nearest_time_index(quasi_T, T[time_idx])
            pinn_at_time = get_pinn_solution(model, T, X, time_idx)

            ax.plot(
                X,
                pinn_at_time,
                "--",
                label="PINN",
            )
            ax.plot(
                X,
                nfd_solution[time_idx, :, 0],
                label="NFD",
            )
            ax.plot(
                quasi_X,
                quasi_solution[quasi_time_idx, :, 0],
                ":",
                label="Квази",
            )
            ax.set_title(f"Время t = {T[time_idx]:g}")
            ax.set_xlabel("x")
            ax.grid(True)
            if col == 0:
                ax.set_ylabel("u(t, x)")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(
            f"N_x = {nx}: сравнение лучшего PINN, NFD и квази-решения; "
            f"параметры PINN: {row['stage']}/{row['case']}; "
            f"относительная ошибка L2 = {row['relative_l2_error']:.3e}"
        )
        fig.tight_layout()
        output = output_dir / f"best_nx_{nx}_pinn_nfd_quasi.png"
        fig.savefig(output, dpi=200)
        plt.close(fig)
        print(f"Best PINN/NFD/Quasi plot saved to {output}")


def make_plots(
    rows: list[dict],
    input_dir: Path,
    manifest: dict,
    output_dir: Path,
    time_moments: list[float],
):
    plot_epochs(rows, output_dir)
    plot_numeric_stage(rows, "width", "model_layers_width", output_dir)
    plot_numeric_stage(rows, "layers", "model_layers_num", output_dir)
    plot_initial_count(rows, output_dir)
    plot_numeric_stage(rows, "collocation_count", "collocation_count", output_dir)
    plot_best_pinn_nfd_quasi(
        input_dir,
        rows,
        load_quasi(input_dir, manifest),
        time_moments,
        output_dir,
    )


def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize and plot sequential Burgers PINN benchmark files."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("benchmark_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_plots"))
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="Re-read HDF5 files even when benchmark_summary.csv exists.",
    )
    parser.add_argument(
        "--comparison-time-moments",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
        help="Time moments for best PINN/NFD/Quasi comparison plots.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArgs(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.input_dir)
    rows = [] if args.recalculate else read_summary(args.input_dir, manifest)
    if not rows:
        rows = calculate_summary(args.input_dir, manifest)

    summary_path = args.output_dir / "benchmark_summary.csv"
    write_summary(rows, summary_path)
    make_plots(
        rows,
        args.input_dir,
        manifest,
        args.output_dir,
        args.comparison_time_moments,
    )
    print(f"Summary saved to {summary_path}")
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
