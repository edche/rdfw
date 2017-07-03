# rdfw
Rank-Drop Frank-Wolfe

Code is currently optimized for sparse gradient.
If gradient is dense, code can still be used but may be suboptimal.

If you wish to use this code for applications other than Matrix
Completion, update the sp_matrix_completion_squared_error.m file to
reflect the objective and gradient calculations.
calc_step_size.m may also need to be updated based upon the optimal step
length for a new objective.
