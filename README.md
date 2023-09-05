This a todo document and most parts are in draft state. Do not take the information here as absolute true

## Benchmark comparison

All the benchmarks are done using [hyperfine](https://lib.rs/crates/hyperfine) with the parameters returned
by `params_100r_140c()`

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

```commandline
$ hyperfine --runs 5 'python nrp_req_as_variable.py'
Benchmark 1: python nrp_req_as_variable.py
  Time (mean ± σ):     32.498 s ±  0.886 s    [User: 33.038 s, System: 2.751 s]
  Range (min … max):   31.786 s … 33.950 s    5 runs
```

### Vectorizing the hot path

The previous implementation uses mostly python's native lists and requires to call an `_evaluate` function
for every solution in every generation.

Pymoo provides the ability to define a vectorized problem, where `_evaluate` is called with a matrix of pop_size rows
and
len_vars columns. This allows to take advantage of numpy functions.

This iteration relies heavily on `apply_along_axis` and the function defined in the previous iteration. This change,
improved performance in about 33.3%. For 500 generations it took ~10sec and only ~6sec to find the first
feasible solution

There is still a lot to do in order to improve performance. Remove `apply_along_axis` calls and replace them
with some linear algebra should be a lot faster

```commandline
$ hyperfine --runs 5 'python nrp_req_as_variable_vectorized.py'
Benchmark 1: python nrp_req_as_variable_vectorized.py
  Time (mean ± σ):     11.261 s ±  0.367 s    [User: 11.828 s, System: 2.556 s]
  Range (min … max):   10.626 s … 11.544 s    5 runs
```
