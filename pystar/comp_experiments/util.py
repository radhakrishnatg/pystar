from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from pystar.core.symbolic_regression import SymbolicRegressionModel


@dataclass
class CommonParams:
    """Dataclass for storing common hyperparameters"""

    operator_list: list
    model_type: str = "bigm"
    depth: int = 5
    selection_criterion: str = "bic"
    penalty_metric: str = "nodes"
    add_cuts: bool = False
    custom_bounds: bool = False
    v_bounds: tuple | None = None
    cst_bounds: tuple = (-100, 100)
    solver: str = "gams:baron"
    max_time: float = 3600.0
    relative_gap: float = 0.0
    absolute_gap: float = 0.0
    log_file_prefix: str = ""
    display_log: bool = False
    use_symbolic_labels: bool = False

    @property
    def fitness_metric(self):
        """Returns the choice of objective function"""
        if self.selection_criterion == "sse":
            return self.selection_criterion

        return f"{self.selection_criterion}_{self.penalty_metric}"

    def get_var_bounds(
        self, train_data: pd.DataFrame | None = None, default: tuple = (-100, 100)
    ):
        """Returns the variable bounds for a given dataset"""
        if self.v_bounds is not None:
            return self.v_bounds

        def_lb, def_ub = default
        if train_data is not None:
            return (
                min(def_lb, train_data.min().min() - 1),
                max(def_ub, train_data.max().max() + 1),
            )

        return (def_lb, def_ub)

    def write_params_to_file(self, filename):
        """Writes the default parameter values to a file"""
        with open(filename, "w") as fp:
            json.dump(self.__dict__, fp, indent=4)

    def solve_model(self, mdl: SymbolicRegressionModel, write_files_path: Path):
        """Solves the symbolic regression model"""
        if "gams" in self.solver:
            return self.solve_model_with_gams(mdl, write_files_path)

        if self.solver == "baron":
            return self.solve_model_with_baron(mdl, write_files_path)

        if self.solver == "gurobi":
            return self.solve_model_with_gurobi(mdl, write_files_path)

        raise ValueError(f"Unrecognized value {self.solver} for solver")

    def solve_model_with_gams(
        self, mdl: SymbolicRegressionModel, write_files_path: Path
    ):
        """Solves the model with solvers in GAMS"""

        solver = pyo.SolverFactory(self.solver.replace("_", ":"))
        logfile = (
            write_files_path / f"{self.log_file_prefix}solver_log_{self.solver}.log"
        )

        results = solver.solve(
            mdl,
            logfile=logfile,
            tee=self.display_log,
            symbolic_solver_labels=self.use_symbolic_labels,
            keepfiles=True,
            tmpdir=str(write_files_path / f"{self.log_file_prefix + self.solver}"),
            add_options=[f"option reslim = {self.max_time};"],  # in seconds
        )

        return {
            "lower_bound": results["Problem"][0]["Lower bound"],
            "upper_bound": results["Problem"][0]["Upper bound"],
            "status": results["solver"][0]["Status"],
            "termination_condition": results["Solver"][0]["Termination condition"],
            "user_time": results["Solver"][0]["User time"],
            "system_time": results["Solver"][0]["System time"],
            "wallclock_time": results["Solver"][0]["Wallclock time"],
        }

    def solve_model_with_baron(
        self, mdl: SymbolicRegressionModel, write_files_path: Path
    ):
        """
        Solves the model using Pyomo/Baron interface
        i.e., standalone Baron solver
        """
        solver = pyo.SolverFactory("baron")
        logfile = (
            write_files_path / f"{self.log_file_prefix}solver_log_{self.solver}.log"
        )

        results = solver.solve(
            mdl,
            solver=self.solver,
            logfile=logfile,
            tee=self.display_log,
            symbolic_solver_labels=self.use_symbolic_labels,
        )

        return {
            "lower_bound": results["Problem"][0]["Lower bound"],
            "upper_bound": results["Problem"][0]["Upper bound"],
            "status": results["solver"][0]["Status"],
            "termination_condition": results["Solver"][0]["Termination condition"],
            "user_time": results["Solver"][0]["Time"],
            "system_time": results["Problem"][0]["Cpu time"],
            "wallclock_time": results["Problem"][0]["Wall time"],
            "num_iterations": results["Problem"][0]["Iterations"],
            "num_nodes": results["Problem"][0]["Node memmax"],
        }

    def solve_model_with_gurobi(
        self, mdl: SymbolicRegressionModel, write_files_path: Path
    ):
        """Solves the model with Gurobi"""
        solver = pyo.SolverFactory("gurobi_persistent")
        solver.set_instance(mdl)
        solver.solve(mdl, tee=True)
        return {}


def get_sr_model(
    train_data: pd.DataFrame,
    params: CommonParams,
    results_dir: Path,
    test_data: pd.DataFrame | None = None,
):
    """Constructs and solves the symbolic regression model for a given dataset"""
    var_bounds = params.get_var_bounds(train_data)
    m = SymbolicRegressionModel(
        data=train_data,
        input_columns=train_data.columns[:-1].to_list(),
        output_column=train_data.columns[-1],
        tree_depth=params.depth,
        operators=params.operator_list,
        var_bounds=var_bounds,
        constant_bounds=params.cst_bounds,
        model_type=params.model_type,
    )
    m.add_objective(params.fitness_metric)

    if params.add_cuts:
        m.add_same_operand_operation_cuts()
        m.add_constant_operation_cuts()
        m.add_associative_operation_cuts()
        m.add_inverse_function_composition_cuts()
        m.add_symmetry_breaking_cuts()
        m.add_implication_cuts()

    results = params.solve_model(m, results_dir)
    results["objective_func_value"] = next(
        m.component_data_objects(pyo.Objective)
    ).expr()
    results["var_bounds"] = var_bounds
    results["cst_values"] = {node: m.constant_val[node].value for node in m.nodes_set}
    results["solution"] = dict(m.get_selected_operators())

    # Check if there are any numerical issues by comparing computed nodal values
    # with the true nodal values
    has_numerical_errors = False
    for sample in m.samples:
        nodal_values = m.samples[sample].compare_node_values()
        nodal_value_errors = (
            nodal_values["Computed Value"] - nodal_values["True Value"]
        ).abs()
        if (nodal_value_errors > 1e-3).any():
            has_numerical_errors = True

    results["has_numerical_errors"] = has_numerical_errors

    expr = m.selected_tree_to_expression()
    results["expression"] = str(expr.sympy_expression)
    results["train_data_sse_model"] = pyo.value(m.sum_square_residual)

    exact_predictions = expr.evaluate_expression(train_data)
    output_data = train_data[train_data.columns[-1]]
    output_var_range = output_data.max() - output_data.min()
    if exact_predictions.isna().any():
        # Error encountered in evaluating the expression.
        # Setting errors to an arbitrary big number
        results["train_data_sse_expression"] = 1e6
    else:
        results["train_data_sse_expression"] = (
            (exact_predictions - output_data) ** 2
        ).sum()

    results["RMSE"] = np.sqrt(results["train_data_sse_expression"] / len(output_data))
    results["NRMSE"] = (
        results["RMSE"] / output_var_range if not np.isclose(output_var_range, 0) else 0
    )

    if test_data is not None:
        exact_predictions = expr.evaluate_expression(test_data)
        if exact_predictions.isna().any():
            # Encountered an error while evaluating the expression
            results["test_data_sse"] = 1e6
        else:
            results["test_data_sse"] = (
                (exact_predictions - test_data[test_data.columns[-1]]) ** 2
            ).sum()

    results["is_sr_prediction_correct"] = (
        1
        if abs(results["train_data_sse_model"] - results["train_data_sse_expression"])
        < 1e-4
        else 0
    )
    results["is_sr_model_good_on_test"] = (
        1 if results.get("test_data_sse", 0) < 1e-3 else 0
    )
    results["law_probably_found"] = (
        results["is_sr_prediction_correct"]
        * results["is_sr_model_good_on_test"]
        * (1 - int(results["has_numerical_errors"]))
    )

    results_filename = f"{params.log_file_prefix}solution_summary_{params.solver}.json"
    with open(results_dir / results_filename, "w") as fp:
        json.dump(results, fp, indent=4)

    return m, results
