from pyomo.core import PositiveReals, Binary, Param, AbstractModel, Set, Var, Constraint, Objective, maximize
from pyomo.environ import *

from knapsack import ITEMS, PROFITS, WEIGHTS, MAXIMUM_CAP

# Source: https://github.com/Pyomo/pyomo/blob/main/examples/doc/samples/scripts/s1/knapsack.py
model = AbstractModel()

model.ITEMS = Set()

model.p = Param(model.ITEMS, within=PositiveReals)

model.w = Param(model.ITEMS, within=PositiveReals)

model.limit = Param(within=PositiveReals)

model.x = Var(model.ITEMS, within=PositiveReals, bounds=(0, 1))


def value_rule(model):
    return (-1) * sum(model.p[i] * model.x[i] for i in model.ITEMS)


model.value = Objective(sense=minimize, rule=value_rule)


def weight_rule(model):
    return sum(model.w[i] * model.x[i] for i in model.ITEMS) <= model.limit


model.weight = Constraint(rule=weight_rule)

instance = model.create_instance(
    data={
        None: {
            'ITEMS': {None: ITEMS},
            'p': {k: v for k, v in zip(ITEMS, PROFITS)},
            'w': {k: v for k, v in zip(ITEMS, WEIGHTS)},
            'limit': {None: MAXIMUM_CAP},

        }
    }
)
print(instance)
solver = SolverFactory('cbc')
res = solver.solve(instance)
print(value(instance.value))
