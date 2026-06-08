import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch

from generate_data import (
    createBoundaryDataFromArrays,
    createInitialCollocationData,
    createModel,
    createPinnSolver,
    createQuasiSolution,
    createTimeGridForSpace,
    parseInitialCount,
    write_dataset,
    write_model,
)


DEFAULT_EPOCHS = (0, 100, 200, 500, 1000, 2000)
DEFAULT_WIDTHS = (2, 4, 8, 16, 24, 32, 40, 64)
DEFAULT_LAYERS = (2, 4, 8, 10, 12, 15)
DEFAULT_INITIAL_COUNTS = ("10", "50", "100", "all")
DEFAULT_COLLOCATION_COUNTS = (100, 200, 500, 1000, 5000, 10000)
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


def has_format(path: Path, expected_format: str) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            return h5.attrs.get("format") == expected_format
    except OSError:
        return False


def attrs_match(path: Path, nx: int, stage: str, case_name: str, config: dict) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("format") != "burgers-benchmark-case":
                return False
            if int(h5.attrs.get("Nx", -1)) != nx:
                return False
            if h5.attrs.get("stage") != stage or h5.attrs.get("name") != case_name:
                return False

            for key, value in serializable_config(config).items():
                if key not in h5.attrs:
                    return False
                if str(h5.attrs[key]) != str(value):
                    return False
    except OSError:
        return False
    return True


def write_quasi(output_dir: Path, args):
    quasi_path = output_dir / "quasi.hdf5"
    if (
        quasi_path.exists()
        and not args.overwrite_quasi
        and has_format(quasi_path, "burgers-benchmark-quasi")
    ):
        print(f"Skipping existing {quasi_path}")
        return

    print(f"Generating {quasi_path}")
    if args.dry_run:
        return

    quasi_t, quasi_x, quasi_solution = createQuasiSolution(args.quasi_space)
    with h5py.File(quasi_path, "w") as h5:
        h5.attrs["format"] = "burgers-benchmark-quasi"
        h5.attrs["quasi_space"] = args.quasi_space
        write_dataset(h5, "T", quasi_t)
        write_dataset(h5, "X", quasi_x)
        write_dataset(h5, "solution", quasi_solution)


def load_quasi(output_dir: Path):
    with h5py.File(output_dir / "quasi.hdf5", "r") as h5:
        return h5["T"][()], h5["X"][()], h5["solution"][()]


def parse_initial_count_label(value):
    if isinstance(value, str):
        return value
    return "all" if value is None else str(value)


def parse_initial_count_value(value):
    if isinstance(value, str):
        return parseInitialCount(value)
    return value


def nx_values(args) -> list[int]:
    if args.nx_values is not None:
        return args.nx_values
    return [args.n_start * 2**idx for idx in range(args.steps)]


def base_config(args) -> dict:
    return {
        "adam_epochs": args.adam_epochs,
        "lbfgs_iters": args.lbfgs_iters,
        "model_layers_width": args.model_layers_width,
        "model_layers_num": args.model_layers_num,
        "initial_count": args.initial_count,
        "boundary_count": args.boundary_count,
        "collocation_count": args.collocation_count,
    }


def normalized_config(config: dict) -> dict:
    result = dict(config)
    result["initial_count"] = parse_initial_count_value(result["initial_count"])
    return result


def serializable_config(config: dict) -> dict:
    result = dict(config)
    result["initial_count"] = parse_initial_count_label(result["initial_count"])
    return result


def stage_candidates(stage: str, selected: dict, args):
    if stage == "epochs":
        for adam_epochs in args.epochs:
            for lbfgs_iters in args.epochs:
                config = dict(selected)
                config.update({"adam_epochs": adam_epochs, "lbfgs_iters": lbfgs_iters})
                yield f"adam_{adam_epochs}_lbfgs_{lbfgs_iters}", config
    elif stage == "width":
        for width in args.widths:
            config = dict(selected)
            config["model_layers_width"] = width
            yield f"width_{width}", config
    elif stage == "layers":
        for layers_num in args.layers:
            config = dict(selected)
            config["model_layers_num"] = layers_num
            yield f"layers_{layers_num}", config
    elif stage == "initial_count":
        for initial_count in args.initial_counts:
            config = dict(selected)
            config["initial_count"] = initial_count
            yield f"initial_{initial_count}", config
    elif stage == "collocation_count":
        for collocation_count in args.collocation_counts:
            config = dict(selected)
            config["collocation_count"] = collocation_count
            yield f"collocation_{collocation_count}", config
    else:
        raise ValueError(f"Unknown benchmark stage: {stage}")


def load_model(model_group: h5py.Group):
    layers_num = int(model_group.attrs["layers_num"])
    layers_width = int(model_group.attrs["layers_width"])
    output_size = int(model_group.attrs["output_size"])
    model = createModel(output_size, layers_num, layers_width)

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


def calculate_error_from_model(model, T, X, quasi_data) -> float:
    quasi_T, quasi_X, quasi_solution = quasi_data
    pinn_solution = get_pinn_grid_solution(model, T, X)
    quasi_on_grid = interpolate_quasi_solution(T, X, quasi_T, quasi_X, quasi_solution)
    return float(relative_l2_error(pinn_solution, quasi_on_grid))


def calculate_error_from_file(output: Path, quasi_data) -> float:
    with h5py.File(output, "r") as h5:
        model = load_model(h5["model"])
        return calculate_error_from_model(model, h5["T"][()], h5["X"][()], quasi_data)


def generate_case(output: Path, nx: int, stage: str, case_name: str, config: dict, args):
    torch.manual_seed(args.seed)
    config = normalized_config(config)
    x_start = np.float64(-1)
    x_end = np.float64(1)
    X = np.linspace(x_start, x_end, nx + 1, dtype=np.float64)
    T = createTimeGridForSpace(X, np.float64(0), np.float64(1))
    boundary_points, boundary_values = createBoundaryDataFromArrays(T, X)
    initial_collocation_points = createInitialCollocationData(boundary_points)

    solver = createPinnSolver(
        T,
        X,
        boundary_points,
        boundary_values,
        config["model_layers_num"],
        config["model_layers_width"],
        config["initial_count"],
        config["boundary_count"],
        config["collocation_count"],
        initial_collocation_points,
    )

    if config["adam_epochs"] > 0:
        solver.trainModel(config["adam_epochs"])
    if config["lbfgs_iters"] > 0:
        solver.trainLBFGS(config["lbfgs_iters"])

    with h5py.File(output, "w") as h5:
        h5.attrs["format"] = "burgers-benchmark-case"
        h5.attrs["stage"] = stage
        h5.attrs["name"] = case_name
        h5.attrs["Nx"] = nx
        attrs = serializable_config(config)
        for key, value in attrs.items():
            h5.attrs[key] = value

        write_dataset(h5, "T", T)
        write_dataset(h5, "X", X)
        write_dataset(
            h5,
            "initial_collocation_points",
            initial_collocation_points.detach().cpu().numpy(),
        )
        write_model(
            h5.create_group("model"),
            solver,
            config["model_layers_num"],
            config["model_layers_width"],
        )

    return T, X, solver.model


def make_row(nx: int, stage: str, case_name: str, output: Path, config: dict, error):
    row = {
        "nx": nx,
        "stage": stage,
        "case": case_name,
        "path": str(output),
        "relative_l2_error": error,
    }
    row.update(serializable_config(config))
    return row


def write_summary(output_dir: Path, rows: list[dict]):
    summary_path = output_dir / "benchmark_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_best_params(output_dir: Path, best_by_nx: dict):
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(best_by_nx, file, indent=2)


def write_manifest(output_dir: Path, args, cases: list[dict], best_by_nx: dict):
    manifest = {
        "format": "burgers-sequential-benchmark-v1",
        "quasi_path": "quasi.hdf5",
        "summary_path": "benchmark_summary.csv",
        "best_params_path": "best_params.json",
        "stages": list(args.stages),
        "nx_values": nx_values(args),
        "common": {
            "quasi_space": args.quasi_space,
            "model_layers_num": args.model_layers_num,
            "model_layers_width": args.model_layers_width,
            "initial_count": args.initial_count,
            "boundary_count": args.boundary_count,
            "collocation_count": args.collocation_count,
            "adam_epochs": args.adam_epochs,
            "lbfgs_iters": args.lbfgs_iters,
            "seed": args.seed,
        },
        "cases": cases,
        "best_by_nx": best_by_nx,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def run_stage(output_dir: Path, nx: int, stage: str, selected: dict, args, quasi_data):
    rows = []
    cases = []
    best_row = None
    best_config = None

    for case_name, config in stage_candidates(stage, selected, args):
        relative_path = Path(f"nx_{nx}") / stage / f"{case_name}.hdf5"
        output = output_dir / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)

        if (
            output.exists()
            and not args.overwrite
            and attrs_match(output, nx, stage, case_name, config)
        ):
            print(f"Skipping existing {output}")
            error = calculate_error_from_file(output, quasi_data)
        else:
            print(f"Generating {output}: {serializable_config(config)}")
            if args.dry_run:
                error = np.nan
            else:
                T, X, model = generate_case(output, nx, stage, case_name, config, args)
                error = calculate_error_from_model(model, T, X, quasi_data)

        row = make_row(nx, stage, case_name, relative_path, config, error)
        rows.append(row)
        cases.append(
            {
                "nx": nx,
                "stage": stage,
                "name": case_name,
                "path": str(relative_path),
                "params": serializable_config(config),
            }
        )
        if not args.dry_run and (best_row is None or error < best_row["relative_l2_error"]):
            best_row = row
            best_config = dict(config)

    if args.dry_run:
        best_config = dict(next(stage_candidates(stage, selected, args))[1])
        best_row = rows[0]

    return rows, cases, best_row, best_config


def run_nx_benchmark(output_dir: Path, nx: int, args):
    quasi_data = None if args.dry_run else load_quasi(output_dir)
    rows = []
    cases = []
    best_by_stage = {}
    selected = base_config(args)

    print(f"Benchmark Nx = {nx}")
    for stage in args.stages:
        stage_rows, stage_cases, best_row, selected = run_stage(
            output_dir, nx, stage, selected, args, quasi_data
        )
        rows.extend(stage_rows)
        cases.extend(stage_cases)
        best_by_stage[stage] = {
            "case": best_row["case"],
            "relative_l2_error": best_row["relative_l2_error"],
            "params": serializable_config(selected),
        }
        print(
            f"Best for Nx={nx}, stage={stage}: "
            f"{best_row['case']} ({best_row['relative_l2_error']:.6e})"
        )

    return nx, rows, cases, best_by_stage


def run_all_nx(output_dir: Path, args):
    values = nx_values(args)
    jobs = min(args.jobs, len(values))
    if jobs <= 1:
        return [run_nx_benchmark(output_dir, nx, args) for nx in values]

    results = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_nx_benchmark, output_dir, nx, args): nx
            for nx in values
        }
        for future in as_completed(futures):
            nx = futures[future]
            result = future.result()
            print(f"Finished Nx = {nx}")
            results.append(result)
    return sorted(results, key=lambda item: values.index(item[0]))


def parseArgs(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate sequential lightweight PINN benchmark files."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_data"))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "epochs",
            "width",
            "layers",
            "initial_count",
            "collocation_count",
        ),
        default=[
            "epochs",
            "width",
            "layers",
            "initial_count",
            "collocation_count",
        ],
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=list(DEFAULT_EPOCHS))
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--initial-counts", nargs="+", default=list(DEFAULT_INITIAL_COUNTS)
    )
    parser.add_argument(
        "--collocation-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COLLOCATION_COUNTS),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-quasi", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--quasi-space", type=int, default=2000)
    parser.add_argument("--model-layers-num", type=int, default=10)
    parser.add_argument("--model-layers-width", type=int, default=64)
    parser.add_argument("--initial-count", default="all")
    parser.add_argument("--boundary-count", type=int, default=200)
    parser.add_argument("--collocation-count", type=int, default=10000)
    parser.add_argument("--adam-epochs", type=int, default=2000)
    parser.add_argument("--lbfgs-iters", type=int, default=2000)
    parser.add_argument("--n-start", type=int, default=100)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--nx-values", type=int, nargs="+", default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("BENCHMARK_JOBS", "1")),
        help="Number of Nx benchmarks to run in parallel.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArgs(argv)
    args.jobs = max(1, args.jobs)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    write_quasi(output_dir, args)
    all_rows = []
    all_cases = []
    best_by_nx = {}

    for nx, rows, cases, best_by_stage in run_all_nx(output_dir, args):
        all_rows.extend(rows)
        all_cases.extend(cases)
        best_by_nx[str(nx)] = best_by_stage
        write_summary(output_dir, all_rows)
        write_best_params(output_dir, best_by_nx)

    write_manifest(output_dir, args, all_cases, best_by_nx)
    write_summary(output_dir, all_rows)
    write_best_params(output_dir, best_by_nx)


if __name__ == "__main__":
    main()
