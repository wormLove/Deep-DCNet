from core.optimizers.least_squares import ActivityOptimizerLeastSquares
from core.optimizers.inverse_matrix import ActivityOptimizerInverseMatrix
from core.optimizers.iterative import IterativeActivityOptimizer

# name -> activity optimizer class. Aliases preserved from the original
# DCNet-V-1.0 DiscriminationModule._build_activity_optimizer.
ACTIVITY_OPTIMIZERS = {
    "least_squares": ActivityOptimizerLeastSquares,
    "ls": ActivityOptimizerLeastSquares,
    "inverse_matrix": ActivityOptimizerInverseMatrix,
    "inverse": ActivityOptimizerInverseMatrix,
    "iterative": IterativeActivityOptimizer,
    "iter": IterativeActivityOptimizer,
}
