This a todo document and most parts are in draft state. Do not take the information here as absolute true

## Benchmark comparison

All the benchmarks are done using [hyperfine](https://lib.rs/crates/hyperfine)
and [kernprof](https://github.com/pyutils/line_profiler) with the parameters returned
by `params_100r_140c()` and running for 500 generations

### One to one translation from MILP NRP

First working version was a one to one translation to see if pymoo works well.
This version could find a feasible result, but it was pretty slow to run. It almost did not use numpy and the hot path
was not vectorized. It took ~33 sec to run 500 generations and in that time it could not find a feasible resoult.
The first feasible result is found at ~68 sec runtime

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
improved performance in about 2.88x. For 500 generations it took ~10sec and only ~6sec to find the first
feasible solution

There is still a lot to do in order to improve performance. Remove `apply_along_axis` calls and replace them
with some linear algebra should be a lot faster

```commandline
$ hyperfine --runs 5 'python nrp_req_as_variable_vectorized.py'  'python nrp_req_as_variable.py'
Benchmark 1: python nrp_req_as_variable_vectorized.py
  Time (mean ± σ):     11.446 s ±  0.207 s    [User: 11.986 s, System: 2.542 s]
  Range (min … max):   11.174 s … 11.735 s    5 runs
 
Benchmark 2: python nrp_req_as_variable.py
  Time (mean ± σ):     32.916 s ±  0.969 s    [User: 33.493 s, System: 2.753 s]
  Range (min … max):   31.533 s … 34.097 s    5 runs
 
Summary
  'python nrp_req_as_variable_vectorized.py' ran
    2.88 ± 0.10 times faster than 'python nrp_req_as_variable.py'
```

### Measuring and optimizing

```commandline
Function: _evaluate at line 127

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   127                                               @profile
   128                                               def _evaluate(self, x: MatrixPopSizeXLenReq, out, *args, **kwargs):
   129                                           
   130                                                   # x : Matrix(pop_size, n_var)
   131       500       3856.3      7.7      0.0          x = x.astype(int)
   132       500    7951199.2  15902.4     25.5          f1 = self._calculate_obj_function_over_matrix(x)
   133                                           
   134       500    6451075.1  12902.2     20.7          g_prereq = self._prereq_constraint_over_matrix(x)
   135       500   14733253.9  29466.5     47.3          g_interes = self._interest_constraint_over_matrix(x)
   136       500    1992304.2   3984.6      6.4          g_cost = self._cost_constraint_over_matrix(x)
   137                                           
   138       500       6349.0     12.7      0.0          out["F"] = np.column_stack((f1,))
   139       500      16932.6     33.9      0.1          out["G"] = np.column_stack((g_prereq, g_interes, g_cost))

```

### Using only implemented requirements (x) as decision variables

By caring only for which requirements is implementing, not using the interest constraint and therefore ignoring `y`
the model reduce it's computational needs and each generation can run faster.

It improves the execution time by 40%.

```commandline
$ hyperfine --runs 5 'python nrp_req_as_variable_vectorized_2.py' 'python nrp_req_as_variable_vectorized_2_all_constraints.py'
Benchmark 1: python nrp_req_as_variable_vectorized_2.py
  Time (mean ± σ):      7.465 s ±  0.226 s    [User: 8.101 s, System: 2.579 s]
  Range (min … max):    7.273 s …  7.802 s    5 runs
 
Benchmark 2: python nrp_req_as_variable_vectorized_2_all_constraints.py
  Time (mean ± σ):     10.543 s ±  0.234 s    [User: 11.126 s, System: 2.668 s]
  Range (min … max):   10.318 s … 10.932 s    5 runs
 
Summary
  'python nrp_req_as_variable_vectorized_2.py' ran
    1.41 ± 0.05 times faster than 'python nrp_req_as_variable_vectorized_2_all_constraints.py'

```

### Using only stakeholders' satisfaction as decision variable

Using the same strategy as before, we can try only using the `y` and ignoring the `x`. This improves the result compared
with the full model but it's slower than the one with only `x`.

It runs 18% slower

```commandline
$ hyperfine --runs 5 'python nrp_req_as_variable_vectorized_2.py' 'python nrp_satisfaction_as_var.py'
Benchmark 1: python nrp_req_as_variable_vectorized_2.py
  Time (mean ± σ):      7.691 s ±  0.157 s    [User: 8.231 s, System: 2.563 s]
  Range (min … max):    7.486 s …  7.869 s    5 runs
 
Benchmark 2: python nrp_satisfaction_as_var.py
  Time (mean ± σ):      9.103 s ±  0.297 s    [User: 9.664 s, System: 2.567 s]
  Range (min … max):    8.800 s …  9.509 s    5 runs
 
Summary
  'python nrp_req_as_variable_vectorized_2.py' ran
    1.18 ± 0.05 times faster than 'python nrp_satisfaction_as_var.py'
```