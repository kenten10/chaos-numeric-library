"""Public exception and warning hierarchy."""


class ChaosNumericsError(Exception):
    """Base class for library-specific exceptions."""


class ValidationError(ChaosNumericsError):
    """Raised when an input violates a public shape, dtype, or value contract."""


class NumericalError(ChaosNumericsError):
    """Raised when numerical failure prevents a meaningful result."""


class ConvergenceError(NumericalError):
    """Raised when strict convergence is required but not achieved."""


class ChaosNumericsWarning(UserWarning):
    """Base class for warnings emitted by the library."""


class NumericalWarning(ChaosNumericsWarning):
    """Warns that a result exists but a numerical diagnostic is unsatisfactory."""


class ConvergenceWarning(NumericalWarning):
    """Warns that a meaningful result did not meet its convergence target."""


class ReproducibilityWarning(ChaosNumericsWarning):
    """Warns that an execution cannot meet the requested reproducibility level."""


__all__ = [
    "ChaosNumericsError",
    "ChaosNumericsWarning",
    "ConvergenceError",
    "ConvergenceWarning",
    "NumericalError",
    "NumericalWarning",
    "ReproducibilityWarning",
    "ValidationError",
]
