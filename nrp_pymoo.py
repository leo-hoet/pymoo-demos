from typing import List, Tuple

import numpy as np
from pymoo.algorithms.soo.nonconvex import ga
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize

from knapsack import BestCandidateCallback, run_and_return_scatter, combine_and_save_scatter, binary_tournament
from model_params import NRPParams, params_100r_140c

RequirementIdx = int
StakeholderIdx = int


class NRP(ElementwiseProblem):
    def __init__(self, params: NRPParams):
        self.len_req = params.len_req
        self.len_customers = params.len_customers
        self.profits = params.profits_per_customer
        self.cost = params.cost_per_req
        self.pre_req_set = params.pre_req_set
        self.interest_set = params.interest_set
        self.max_allowed_cost = params.max_allowed_cost

        n_var = self.len_req + self.len_customers

        xl = np.zeros(n_var)
        xu = np.ones(n_var)

        cost_constraints = 1

        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=len(self.pre_req_set) + len(self.interest_set) + cost_constraints,
            # extra one for max const constraint
            xl=xl,
            xu=xu,
        )

    def _get_xs_ys(self, x):
        xs = x[:self.len_req]  # 1 if req is implemented
        ys = x[self.len_req:(self.len_req + self.len_customers)]  # 1 if customer is satisfied
        return xs, ys

    def _calculate_obj_function(self, x) -> float:
        xs, ys = self._get_xs_ys(x)

        f1 = 0

        # calculate profit
        for customer_is_satisfied, profit_customer_satisfied in zip(ys, self.profits):
            f1 += customer_is_satisfied * profit_customer_satisfied

        # calculate costs
        for req_is_implemented, implementation_cost in zip(xs, self.cost):
            f1 -= req_is_implemented * implementation_cost
        f1 = (-1) * f1  # pymoo only accepts minimization O.F. So, multiply by -1
        return f1

    def _prereq_constraint(self, x) -> List[float]:
        xs, _ = self._get_xs_ys(x)
        result = []
        for (i, j) in self.pre_req_set:
            pre_req_not_violated = x[i] + x[j]
            result.append(pre_req_not_violated)
        return result

    def _interest_constraint(self, x) -> List[float]:
        xs, ys = self._get_xs_ys(x)
        result = []
        for i, k in self.interest_set:
            stakeholder_i_interest_is_satisfied = -ys[i] + xs[k]
            result.append(stakeholder_i_interest_is_satisfied)
        return result

    def _cost_constraint(self, x) -> float:
        xs, _ = self._get_xs_ys(x)
        return np.dot(xs, self.cost) - self.max_allowed_cost

    def _evaluate(self, x, out, *args, **kwargs):
        x = [1 if i else 0 for i in x]
        f1 = self._calculate_obj_function(x)

        g_prereq = self._prereq_constraint(x)
        g_interes = self._interest_constraint(x)
        g_cost = self._cost_constraint(x)

        out["F"] = [f1]
        out["G"] = g_prereq + g_interes + [g_cost]


def run_get_data():
    # Writes a csv with
    # num_run, iteration, fo

    raise NotImplementedError()


def main():
    params = params_100r_140c()
    nrp = NRP(params)
    algol = ga.GA(
        sampling=BinaryRandomSampling(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
        pop_size=100
    )

    res = minimize(
        problem=nrp,
        algorithm=algol,
        termination=('n_gen', 300),
        verbose=False,
        selection=RandomSelection(),
        crossover=SBX(),
        callback=BestCandidateCallback(),
    )

    selections = [RandomSelection(), TournamentSelection(pressure=2, func_comp=binary_tournament)]
    crossovers = [SBX(), UniformCrossover(prob=1.0), TwoPointCrossover()]

    traces = []
    for selection in selections:
        for crossover in crossovers:
            trace = run_and_return_scatter(
                problem=nrp,
                algol=algol,
                selection=selection,
                crossover=crossover,
                n_gen=50,
            )
            traces.append(trace)
    combine_and_save_scatter(traces, optimum_value=params.fo_optimum)

    # bests: List[float] = res.algorithm.callback.data['best']
    # print("Algorithm used: ", res.algorithm)
    # print("Time :", res.exec_time)
    # print("Design space values X: ", res.X)
    #  print("Objective space values F: ", res.F)
    # print("Constraint  values G: ", res.G)
    # print("Aggregated constraint violation CV: ", res.CV)
    # print("Final population: ", res.pop.get("X"))


if __name__ == "__main__":
    main()

# use this to display uncertantly https://plotly.com/python/continuous-error-bars/
