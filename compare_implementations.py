import multiprocessing as mp

from pymoo.algorithms.soo.nonconvex import ga
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.rnd import RandomSelection

from knapsack import run_and_return_scatter, combine_and_save_scatter
from model_params import params_100r_140c
from nrp_pymoo import NRP
from nrp_req_as_variable import NRPReqAsVariable


def run_scatter_wrapper(kwargs: dict):
    return run_and_return_scatter(**kwargs)


def main():
    params = params_100r_140c()
    nrp_req_as_x = NRPReqAsVariable(params)
    nrp_full = NRP(params)

    algol = ga.GA(
        sampling=BinaryRandomSampling(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
        pop_size=100
    )
    selection = RandomSelection()
    crossover = SBX()

    n_gen = 1000

    args = [
        {
            'problem': nrp_full,
            'algol': algol,
            'selection': selection,
            "crossover": crossover,
            "n_gen": n_gen,
            "pop_dispersion": False
        },
        {
            'problem': nrp_req_as_x,
            'algol': algol,
            'selection': selection,
            "crossover": crossover,
            "n_gen": n_gen,
            "pop_dispersion": False
        },
    ]
    with mp.Pool(2) as p:
        results = p.map(run_scatter_wrapper, args)

    traces = [r[0] for r in results]
    combine_and_save_scatter(traces, optimum_value=params.fo_optimum)


if __name__ == "__main__":
    main()
