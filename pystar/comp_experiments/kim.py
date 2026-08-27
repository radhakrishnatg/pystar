from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
from pystar.comp_experiments.util import CommonParams, get_sr_model

DATASET_DIR = Path(__file__).parent.parent.parent.parent / "kim-sr" / "data"
DATASETS = [entry.name for entry in DATASET_DIR.iterdir() if entry.is_dir()]

RESULTS_DIR = Path(__file__).parent / "results_kim"
RESULTS_DIR.mkdir(exist_ok=True)

run_mode = "parallel"

cp = CommonParams(
    operator_list=["sum", "diff", "mult", "div", "sqrt", "square", "exp", "log"],
    selection_criterion="sse",
    solver="gams_baron",  # ["gams_baron", "gams_scip", "baron", "gurobi"]
    max_time=100,
    log_file_prefix="fts_",
)
cp.write_params_to_file(filename=RESULTS_DIR / "params.json")


def run_single_case(ds: str):
    """
    Creates a function to run a single case. This function is needed
    to parallelize the runs
    """
    print(f"Solving the problem for dataset {ds}")
    train_data = pd.read_csv(
        DATASET_DIR / ds / "trndata.txt", delimiter="\t", header=None
    )
    test_data = pd.read_csv(
        DATASET_DIR / ds / "tstdata.txt", delimiter="\t", header=None
    )
    # data.columns = [f"x{i+1}" for i in range(len(data.columns))]
    res_dir = RESULTS_DIR / ds
    res_dir.mkdir(exist_ok=True)

    get_sr_model(
        train_data=train_data, params=cp, results_dir=res_dir, test_data=test_data
    )


if run_mode == "parallel":
    # Runs the code in parallel mode
    Parallel(n_jobs=-1)(delayed(run_single_case)(ds) for ds in DATASETS)
else:
    for data_set in DATASETS:
        run_single_case(data_set)
