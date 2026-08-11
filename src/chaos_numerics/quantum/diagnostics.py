"""Dynamical-instability diagnostics: Loschmidt echo and out-of-time-order correlator.

Both routines answer the same question from opposite ends. The Loschmidt echo
measures how fast two *slightly different* Hamiltonians pull the *same* state
apart; the OTOC measures how fast *one* Hamiltonian spreads a local operator
until it stops commuting with a second one. They are the two standard
finite-dimensional stand-ins for the classical Lyapunov instability, which has
no direct quantum analogue because unitary evolution preserves every overlap.

``weyl_translations`` is exported because :func:`otoc` defaults to it and because
a caller who wants a different operator pair needs the shipped one to compare
against.
"""

from __future__ import annotations

import numpy as np

from chaos_numerics.core import (
    AnalysisResult,
    ExperimentMetadata,
    NumericalError,
    QuantumMap,
    ValidationError,
)
from chaos_numerics.core._validation import as_complex_array, validate_protocol
from chaos_numerics.core.types import ArrayLike, ComplexArray
from chaos_numerics.quantum.evolution import EvolutionMethod, QuantumEvolution, evolve
from chaos_numerics.quantum.states import BoundaryPhases, quantum_state
from chaos_numerics.quantum.unitary import (
    DEFAULT_DENSE_LIMIT,
    _dense_representation,
    _dense_unitarity_defect,
)

_UNITARITY_TOLERANCE = 1e-12
"""Largest unitarity defect :func:`otoc` will accept, matching ``eigenstates``.

``otoc`` builds ``A(t) = U^(-t) A U^t`` by taking the adjoint of ``U`` as its
inverse, which is only correct for a unitary ``U``. Without this guard a
non-unitary model produced a smooth, plausible, decaying correlator with no
exception and no warning -- measured on a model whose dense form is ``0.9 * I``
(defect 0.19): ``C(t) = [2.0, 1.312, 0.861, 0.565]``, while ``eigenstates`` and
``evolve`` both refuse the same model. The defect was recorded in the metadata,
which only helps a caller who thinks to read it."""


def weyl_translations(
    dimension: int,
    *,
    boundary_phases: BoundaryPhases | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    r"""Return the ``(T_q, T_p)`` Weyl translation pair on the ``N``-point torus.

    Neither ``q`` nor ``p`` is a well-defined operator on a torus -- both
    coordinates are periodic, so the linear functions that represent them jump
    at the cut and the would-be operators are not even continuous, let alone
    self-adjoint. The Weyl (clock-and-shift) translations are the bounded
    unitary replacements that generate the same algebra:

    * ``T_q = diag(exp(2 pi i q_j))`` with ``q_j = (j + alpha) / N``. It is the
      multiplication operator by ``exp(2 pi i q)``, the lowest non-trivial
      Fourier mode of the position circle.
    * ``T_p`` is the cyclic shift ``j -> j + 1``, with the single wrap-around
      entry carrying ``exp(2 pi i beta)`` so that the momentum twist of the
      quasi-periodic boundary condition is respected.

    ``alpha`` and ``beta`` are ``boundary_phases.position`` and
    ``boundary_phases.momentum`` in turns; the default is the periodic
    ``(0, 0)``.

    Both matrices are exactly unitary and satisfy the Weyl commutation relation

    ``T_q T_p = exp(2 pi i / N) T_p T_q``

    Measured over ``N`` in ``{2, 3, 8, 16, 64, 128, 257, 512}``, worst case over
    that set:

    ================== ======================== ============================
    ``(alpha, beta)``  ``||T.H T - I||_F/sqrt N`` Weyl residual / ``sqrt N``
    ================== ======================== ============================
    ``(0, 0)``         ``6.66e-17``             ``4.28e-16``
    ``(1/2, 1/2)``     ``8.33e-17``             ``5.39e-16``
    ``(0.25, 0.13)``   ``7.73e-17``             ``3.41e-16``
    ================== ======================== ============================

    with the residual measured as
    ``||T_q T_p - exp(2 pi i / N) T_p T_q||_F / sqrt(N)``. Both columns are pure
    rounding in the complex exponential: the shift matrix is exact 0/1 apart
    from its single wrap phase, so the only inexact factor anywhere is
    ``exp(2 pi i (j + alpha) / N)``.

    The returned arrays are read-only, like every other array the library hands
    out. Call ``.copy()`` for a writable one.
    """
    size = _positive_int(dimension, name="dimension")
    phases = BoundaryPhases() if boundary_phases is None else boundary_phases
    if not isinstance(phases, BoundaryPhases):
        raise ValidationError("boundary_phases must be a BoundaryPhases instance or None")
    indices = np.arange(size)
    positions = (indices.astype(np.float64) + phases.position) / size
    clock = np.diag(np.exp(2j * np.pi * positions)).astype(np.complex128)
    shift = np.zeros((size, size), dtype=np.complex128)
    shift[(indices + 1) % size, indices] = 1.0
    shift[0, size - 1] = np.exp(2j * np.pi * phases.momentum)
    clock.setflags(write=False)
    shift.setflags(write=False)
    return clock, shift


def loschmidt_echo(
    reference: QuantumMap,
    perturbed: QuantumMap,
    initial_state: ArrayLike,
    *,
    steps: int,
    method: EvolutionMethod = "auto",
    norm_tolerance: float = 5e-11,
    dense_limit: int = DEFAULT_DENSE_LIMIT,
) -> AnalysisResult:
    r"""Return ``M(t) = |<psi| U_b^(-t) U_a^t |psi>|**2`` for ``t = 0 .. steps``.

    ``reference`` is ``U_a`` and ``perturbed`` is ``U_b``. Taking two models
    rather than one model and a perturbation size is deliberate: the echo is a
    comparison, and which two operators are being compared is exactly the thing
    a reader of the result needs to know. Nothing about the pair is assumed
    beyond a shared :attr:`~chaos_numerics.core.QuantumMap.dimension`, which is
    validated; the two may be different classes.

    **Batched initial states are supported.** ``initial_state`` may carry any
    number of leading batch axes, and ``values`` then has shape
    ``(*batch, steps + 1)``. This costs nothing to support because
    :func:`~chaos_numerics.quantum.evolve` is already batched, and a decay rate
    averaged over initial conditions is the usual way the echo is reported.

    **The echo decays faster the larger the perturbation.** Measured on
    :class:`~chaos_numerics.quantum.KickedRotor` at ``N = 256``, both models
    differing only in ``kick_strength``, starting from the coherent state at
    ``(q, p) = (0.5, 0.0)``, reading ``M(20)``:

    ========= ========== ========== ========== ========== ========== ==========
    ``K``     ``1e-4``   ``3e-4``   ``1e-3``   ``3e-3``   ``1e-2``   ``3e-2``
    ========= ========== ========== ========== ========== ========== ==========
    ``1.0``   1.000000   0.999998   0.999983   0.999843   0.998252   0.984771
    ``5.0``   0.999589   0.996300   0.959453   0.679165   0.020998   0.004536
    ``10.0``  0.999746   0.997712   0.974829   0.793635   0.073779   0.031860
    ========= ========== ========== ========== ========== ========== ==========

    Every row is strictly decreasing, which is the defining property and is what
    the test suite pins.

    **A regular phase space and a chaotic one are separated by orders of
    magnitude.** Same setup, perturbation fixed at ``1e-2``:

    ========= =============== ===============
    ``K``     ``M(20)``       ``M(40)``
    ========= =============== ===============
    0.2       0.976508        9.23e-01
    0.5       0.997523        9.88e-01
    1.0       0.998252        9.92e-01
    2.0       0.993132        9.74e-01
    5.0       0.020998        1.31e-03
    10.0      0.073779        5.86e-03
    ========= =============== ===============

    ``M(40)`` falls by nearly three orders of magnitude between the last KAM
    torus (``K_c = 0.9716``) and the globally chaotic regime, on an unchanged
    perturbation. Two honest caveats. The ordering is not monotone in ``K``:
    ``K = 0.2`` decays *more* than ``K = 0.5``, because the coherent state used
    here sits in a large island at ``K = 0.5`` and in a much flatter, more
    spread-out landscape at ``K = 0.2``, and ``K = 10`` decays *less* than
    ``K = 5``, because ``delta K / hbar_eff`` is what the echo actually
    responds to and the golden-rule regime is not monotone in it either. The
    separation between the two *regimes* is the diagnostic; the ordering inside
    a regime is not.

    ``M(0)`` is ``1`` by construction. It is *bit-exactly* ``1.0`` when the
    initial state has exactly representable amplitudes -- a basis state, for
    instance -- and otherwise equals ``1`` to the rounding of one normalization,
    measured worst error ``2.2e-16`` over 200 random states at ``N = 128``. Two
    identical models give ``M(t) = 1`` to ``2.7e-14`` over 50 steps, and
    ``M(t)`` may exceed ``1`` by up to ``1.8e-15``. The values are deliberately
    **not** clipped into ``[0, 1]``: a clipped diagnostic hides exactly the
    numerical drift it should be reporting.

    ``method``, ``norm_tolerance``, and ``dense_limit`` are forwarded unchanged
    to :func:`~chaos_numerics.quantum.evolve` for **both** models, so a norm
    drift in either one raises :class:`~chaos_numerics.core.NumericalError`
    there rather than silently distorting the echo.

    Memory is ``2 * (steps + 1) * N`` complex numbers, because both histories
    are materialized. At ``N = 512`` and ``steps = 1000`` that is 16 MiB.
    """
    validate_protocol(reference, QuantumMap, name="reference")
    validate_protocol(perturbed, QuantumMap, name="perturbed")
    dimension = int(reference.dimension)
    if int(perturbed.dimension) != dimension:
        raise ValidationError(
            "reference and perturbed models must share one Hilbert-space dimension; got "
            f"{dimension} and {int(perturbed.dimension)}"
        )
    state = quantum_state(initial_state, dimension=dimension, normalized=True)
    forward = evolve(
        reference,
        state,
        steps=steps,
        method=method,
        return_history=True,
        include_initial=True,
        norm_tolerance=norm_tolerance,
        dense_limit=dense_limit,
    )
    comparison = evolve(
        perturbed,
        state,
        steps=steps,
        method=method,
        return_history=True,
        include_initial=True,
        norm_tolerance=norm_tolerance,
        dense_limit=dense_limit,
    )
    overlaps = np.einsum(
        "...ti,...ti->...t", _requested_history(comparison).conj(), _requested_history(forward)
    )
    values = np.asarray(np.abs(overlaps) ** 2, dtype=np.float64)
    metadata = ExperimentMetadata(
        parameters={
            "dimension": dimension,
            "steps": int(values.shape[-1]) - 1,
            "method": method,
            "batch_shape": list(values.shape[:-1]),
            "reference": _model_parameters(reference),
            "perturbed": _model_parameters(perturbed),
            "norm_tolerance": float(norm_tolerance),
            "dense_limit": int(dense_limit),
            "initial_echo": float(np.max(values[..., 0])),
            "final_echo": float(np.mean(values[..., -1])),
            "definition": "M(t) = |<psi| U_perturbed^(-t) U_reference^t |psi>|**2",
        },
        precision="complex128",
    )
    return AnalysisResult("loschmidt_echo", values, metadata=metadata)


def otoc(
    model: QuantumMap,
    *,
    steps: int,
    operator_a: ArrayLike | None = None,
    operator_b: ArrayLike | None = None,
    state: ArrayLike | None = None,
    dense_limit: int = DEFAULT_DENSE_LIMIT,
) -> AnalysisResult:
    r"""Return ``C(t) = <|[A(t), B]|**2>`` with ``A(t) = U^(-t) A U^t``.

    ``<X>`` is the infinite-temperature average ``Tr[X] / N`` unless ``state``
    is given, in which case it is the pure-state expectation
    ``<psi|X|psi>``. The infinite-temperature default is the operator-level
    statement -- it depends on ``A``, ``B``, and ``U`` and on nothing else -- and
    is what makes the number comparable between models. The pure-state variant
    is offered because a wave packet placed in one phase-space region is how the
    quantity is usually *seen*, and it is one extra matrix-vector product.

    ``operator_a`` and ``operator_b`` are arbitrary dense ``(N, N)`` matrices.
    They default to the Weyl translation pair ``(T_q, T_p)`` of
    :func:`weyl_translations`, taken at the model's own ``boundary_phases`` when
    it publishes them; see that function for why ``q`` and ``p`` themselves are
    not available on a torus.

    **``C(0)`` has a closed form for the default pair.** From the Weyl relation
    ``T_q T_p = e^(2 pi i / N) T_p T_q``,

    ``[T_q, T_p] = (e^(2 pi i / N) - 1) T_p T_q``

    and ``T_p T_q`` is unitary, so

    ``C(0) = |e^(2 pi i / N) - 1|**2 = 4 sin(pi / N)**2``

    exactly, at infinite temperature *and* for every pure state, because
    ``[A, B].H [A, B]`` is a multiple of the identity. Measured absolute error
    against ``4 sin(pi / N)**2`` over ``N`` in ``{4, 8, 16, 32, 64, 128, 256}``:
    worst ``2.22e-16``. Taking ``A = B`` instead gives ``C(0) = 0.0``
    bit-exactly, since the commutator is the zero matrix.

    **The integrable and chaotic regimes separate by four orders of
    magnitude.** :class:`~chaos_numerics.quantum.KickedRotor` at ``N = 512``
    (``hbar_eff = 0.0123``) with the default operator pair, against the
    phase-space-averaged classical Lyapunov exponent ``lambda`` -- the mean over
    16 uniformly drawn standard-map initial conditions from
    :func:`~chaos_numerics.classical.largest_lyapunov_exponent`, which is the
    right comparison for an infinite-temperature average and not the exponent of
    one arbitrary orbit:

    ======= ========== ========== ============== =========== ============ =========
    ``K``   ``lambda`` ``2 lam``  rate ``1..4``   ratio       ``C(4)/C(0)`` ``C_sat``
    ======= ========== ========== ============== =========== ============ =========
    0.0     0.0002     0.0003     ``-0.0000``     --          1.0          0.0002
    0.5     0.0001     0.0003     0.5179          --          5.3          0.0203
    1.0     0.0585     0.1171     0.9369          8.00        24.9         0.1575
    2.0     0.3696     0.7392     1.4915          2.02        263.6        1.2250
    5.0     0.9688     1.9376     2.1186          1.09        6727.0       1.9404
    10.0    1.6211     3.2422     1.7794          0.55        11760.1      1.9997
    ======= ========== ========== ============== =========== ============ =========

    ``C_sat`` is the mean over ``20 <= t <= 24``; the ratio column is the fitted
    rate over ``2 lambda`` and is left blank where ``lambda`` is consistent with
    zero. At ``K = 0`` the correlator is **exactly flat** -- measured
    ``max |C(t) - C(0)| = 4.2e-18`` over 40 steps -- because the free rotation
    only multiplies ``T_q`` by phases and never changes ``|[T_q(t), T_p]|``.
    That is the integrable baseline, and the growth at ``K >= 1`` is measured
    against it.

    **The fitted rate is not ``2 lambda``, and it is reported rather than
    fitted away.** The ratio runs 8.00, 2.02, 1.09, 0.55 as ``K`` goes 1, 2, 5,
    10, so it crosses 1 rather than converging to it. Three effects, all of them
    physics rather than bugs:

    * ``C(t)`` measures ``<(d p(t) / d q(0))**2>``, the *second moment* of the
      stability multiplier, so its growth rate is the order-2 generalized
      Lyapunov exponent, which is ``>= 2 lambda`` whenever the finite-time
      exponent fluctuates. That is the overshoot at ``K = 1`` and ``K = 2``.
    * The Ehrenfest time ``t_E ~ ln(1 / hbar_eff) / lambda`` is only ``2.7``
      steps at ``K = 10`` and ``N = 512``. The fit window is then longer than
      the exponential regime it is fitting, and saturation pulls the rate down.
      That is the undershoot at ``K = 10``.
    * ``T_q`` and ``T_p`` are bounded unitaries, not the unbounded ``q`` and
      ``p``, so ``C`` stops at ``2`` rather than continuing. Measured at
      ``N = 128``, ``K = 10``: maximum ``2.0426`` over ``t <= 200``, minimum
      ``1.9557`` for ``t >= 12``, time-average ``1.9946`` over
      ``100 <= t <= 200``. The hard bound is ``4``, from
      ``||[A, B]|| <= 2 ||A|| ||B||``.

    Widening the semiclassical window is not available here: ``hbar_eff`` is
    tied to ``N`` by the torus quantization, and raising ``N`` costs
    ``O(N**3)``. **Use the ratio between regimes, not the absolute rate.**

    **The model must be unitary**, to the same ``1e-12`` defect tolerance
    :func:`~chaos_numerics.quantum.eigenstates` uses, and a model that is not
    raises :class:`~chaos_numerics.core.NumericalError`. The recursion inverts the
    propagator by taking its adjoint, which is the inverse only for a unitary
    operator; without the guard a non-unitary model returned a smooth decaying
    correlator with no warning at all.

    **Cost.** The dense path is the only one, because ``A(t)`` is a matrix and
    there is nothing for a matrix-free algorithm to act on. Each step is four
    ``O(N**3)`` products and ``dense_limit`` guards materializing the model
    exactly as it does elsewhere. Measured: 0.04 s at ``N = 256``,
    ``steps = 24``; 0.11 s at ``N = 512``, ``steps = 8``.
    """
    validate_protocol(model, QuantumMap, name="model")
    count = _nonnegative_int(steps, name="steps")
    dimension = int(model.dimension)
    unitary = _dense_representation(model, dense_limit=dense_limit)
    defect = _dense_unitarity_defect(unitary)
    if defect > _UNITARITY_TOLERANCE:
        raise NumericalError(
            f"unitary defect {defect} exceeds otoc tolerance {_UNITARITY_TOLERANCE}; "
            "the Heisenberg recursion inverts the propagator by taking its adjoint, "
            "which is only the inverse for a unitary operator"
        )
    adjoint = unitary.conj().T
    defaults = weyl_translations(dimension, boundary_phases=_model_boundary_phases(model))
    heisenberg = (
        defaults[0].copy()
        if operator_a is None
        else _square_operator(operator_a, dimension=dimension, name="operator_a")
    )
    other = (
        defaults[1]
        if operator_b is None
        else _square_operator(operator_b, dimension=dimension, name="operator_b")
    )
    vector: ComplexArray | None = None
    if state is not None:
        vector = quantum_state(state, dimension=dimension, normalized=True)
        if vector.ndim != 1:
            raise ValidationError(
                f"state must be a single ({dimension},) vector, not a batch; got {vector.shape}"
            )

    values = np.empty(count + 1, dtype=np.float64)
    for index in range(count + 1):
        commutator = heisenberg @ other - other @ heisenberg
        if vector is None:
            values[index] = float(np.sum(np.abs(commutator) ** 2)) / dimension
        else:
            applied = commutator @ vector
            values[index] = float(np.vdot(applied, applied).real)
        if index < count:
            heisenberg = adjoint @ heisenberg @ unitary

    metadata = ExperimentMetadata(
        parameters={
            "dimension": dimension,
            "steps": count,
            "expectation": "infinite_temperature" if vector is None else "pure_state",
            "operator_a": "weyl_position_translation" if operator_a is None else "caller_supplied",
            "operator_b": "weyl_momentum_translation" if operator_b is None else "caller_supplied",
            "dense_limit": int(dense_limit),
            "unitarity_defect": _dense_unitarity_defect(unitary),
            "model": _model_parameters(model),
            "initial_value": float(values[0]),
            "maximum_value": float(np.max(values)),
            "definition": "C(t) = <|[A(t), B]|**2> with A(t) = U^(-t) A U^t",
        },
        precision="complex128",
    )
    return AnalysisResult("otoc", values, metadata=metadata)


def _requested_history(evolution: QuantumEvolution) -> ComplexArray:
    """Return the history of an evolution that was run with ``return_history=True``.

    ``QuantumEvolution.history`` is ``None`` whenever the history was not asked
    for, which is the right type for the public container but not something a
    caller that always asks for it can narrow. Raising here rather than asserting
    keeps the failure a library error instead of vanishing under ``-O``.
    """
    if evolution.history is None:
        raise NumericalError(  # pragma: no cover - unreachable via the public callers
            "evolve() was asked for a history and returned none"
        )
    return evolution.history


def _model_parameters(model: QuantumMap) -> dict[str, object]:
    """Return the model's JSON parameters, or just its type when it has none."""
    parameters = getattr(model, "parameters", None)
    if isinstance(parameters, dict):
        return {"model": type(model).__name__, **parameters}
    return {"model": type(model).__name__, "dimension": int(model.dimension)}


def _model_boundary_phases(model: QuantumMap) -> BoundaryPhases:
    phases = getattr(model, "boundary_phases", None)
    return phases if isinstance(phases, BoundaryPhases) else BoundaryPhases()


def _square_operator(value: ArrayLike, *, dimension: int, name: str) -> ComplexArray:
    operator = as_complex_array(value, name=name, ndim=2, copy=True)
    if operator.shape != (dimension, dimension):
        raise ValidationError(
            f"{name} must have shape ({dimension}, {dimension}); got {operator.shape}"
        )
    return operator


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValidationError(f"{name} must be positive; got 0")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValidationError(f"{name} must be non-negative; got {result}")
    return result


__all__ = ["loschmidt_echo", "otoc", "weyl_translations"]
