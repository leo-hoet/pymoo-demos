This a todo document and most parts are in draft state. Do not take the information here as absolute true

## Benchmark comparison

### One to one translation from MILP NRP

First working version was a one to one translation to see if pymoo works well.
This version could find a feasible result but it was pretty slow to run. It almost did not use numpy and the hot path
was
not vectorized

### Only requirements as decisions variables

The problem does not need the variables `y` (stakeholder satisfaction). These variables
can be derived from `x`. By doing this, the model is simplified and it is easier to find a
feasible solution.

With this change, we are able to run 500 generations in about ~32 sec. Finding the first
feasible solution at ~23sec.

### Vectorizing the hot path

The previous implementation uses mostly python's native lists and requires to call an `_evaluate` function
for every solution in every generation.

Pymoo provides the ability to define a vectorized problem, where `_evaluate` is called with a matrix of pop_size rows
and
len_vars columns. This allows to take advantage of numpy functions.

This iteration relies heavly on `apply_along_axis` and the function defined in the previous iteration. This change,
improved performance in about 33.3%. For 500 generations it took ~10sec and only ~6sec to find the first
feasible solution


