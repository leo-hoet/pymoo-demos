from collections import defaultdict
from typing import Dict, List

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

from knapsack import binary_tournament, run_and_return_scatter, combine_and_save_scatter
from model_params import NRPParams, StakeholderIdx, RequirementIdx, params_100r_140c


class NRPReqAsVariable(ElementwiseProblem):
    def __init__(self, params: NRPParams):
        self.len_req = params.len_req
        self.len_customers = params.len_customers
        self.profits = params.profits_per_customer
        self.cost = params.cost_per_req
        self.pre_req_set = params.pre_req_set
        self.interest_set = params.interest_set
        self.max_allowed_cost = params.max_allowed_cost

        self.interest_map: Dict[StakeholderIdx, List[RequirementIdx]] = defaultdict(list)
        for (stakeholder_idx, req_idx) in self.interest_set:
            self.interest_map[stakeholder_idx].append(req_idx)

        n_var = self.len_req

        xl = np.zeros(n_var)
        xu = np.ones(n_var)

        cost_constraints = 1

        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=len(self.pre_req_set) + len(self.interest_set) + cost_constraints,
            xl=xl,
            xu=xu,
        )

    def get_ys_from_xs(self, x) -> List[float]:
        ys = [0] * self.len_customers
        for stakeholder_idx, list_req_idx in self.interest_map.items():
            for idx in list_req_idx:
                if x[idx] == 0:
                    break
                ys[stakeholder_idx] = 1
        return ys

    def _get_xs_ys(self, x):
        ys = self.get_ys_from_xs(x)
        return x, ys

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
            pre_req_not_violated = -x[i] + x[j]
            result.append(pre_req_not_violated)
        return result

    def _interest_constraint(self, x) -> List[float]:
        xs, ys = self._get_xs_ys(x)
        result = []
        for i, k in self.interest_set:
            stakeholder_i_interest_is_satisfied = ys[i] - xs[k]
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


def main():
    params = params_100r_140c()
    nrp = NRPReqAsVariable(params)
    algol = ga.GA(
        sampling=BinaryRandomSampling(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
        pop_size=100
    )

    selections = [RandomSelection(), TournamentSelection(pressure=2, func_comp=binary_tournament)]
    crossovers = [SBX(), UniformCrossover(prob=1.0), TwoPointCrossover()]

    selections = selections[:1]
    crossovers = crossovers[:1]

    traces = []
    for selection in selections:
        for crossover in crossovers:
            traces += run_and_return_scatter(
                problem=nrp,
                algol=algol,
                selection=selection,
                crossover=crossover,
                n_gen=500,
            )
    combine_and_save_scatter(traces, optimum_value=params.fo_optimum)


if __name__ == "__main__":
    main()
