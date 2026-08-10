"""Public exception and warning hierarchy."""


class ChaosNumericsError(Exception):
    """Base class for library-specific exceptions."""


class ValidationError(ChaosNumericsError, ValueError):
    """Raised when an input violates a public shape, dtype, or value contract.

    Also a :class:`ValueError` so that ``except ValueError`` in downstream code
    keeps catching rejected inputs, which is what callers expect from a numeric
    library.
    """


class NumericalError(ChaosNumericsError, ArithmeticError):
    """Raised when numerical failure prevents a meaningful result.

    Also an :class:`ArithmeticError` so that generic numerical error handling
    outside this library still applies.
    """


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
