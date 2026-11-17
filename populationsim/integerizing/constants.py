STATUS_OPTIMAL = "OPTIMAL"
STATUS_FEASIBLE = "FEASIBLE"
STATUS_SUCCESS = [STATUS_OPTIMAL, STATUS_FEASIBLE]

# 'CBC', 'GLPK_MI', 'ECOS_BB'
CVX_SOLVER = "GLPK_MI"

# Order of vectorization for cvxpy
# 'C' for C-style row-major order, 'F' for Fortran-style column-major order
# Note: cvxpy is deprecating 'F' order, so we use 'C' order.
ORDER = "C"

# fork (INTEGERIZER_ROUNDING: controlled): names of the LP solvers of the
# controlled rounding, read by bs_options and controlled_lp (which documents
# them); scipy.optimize.linprog with HiGHS (dual simplex, interior point) and
# OR-Tools pywraplp with Clp, Glop and CBC
CONTROLLED_SOLVERS = (
    "highs-ds",
    "highs-ipm",
    "clp-dual",
    "clp-primal",
    "clp-barrier",
    "glop-dual",
    "glop-primal",
    "cbc",
)
