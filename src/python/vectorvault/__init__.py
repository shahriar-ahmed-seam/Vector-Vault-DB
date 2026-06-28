"""Vector-Vault-DB Python interface.

This package wraps the compiled native extension ``_vectorvault`` (built from
the C++ core engine via pybind11) and re-exports the public engine surface:
the :class:`Engine`/:class:`Collection` operations, the
:class:`DistanceMetric`, :class:`IndexType`, and :class:`CollectionInfo`
types, and the :class:`VectorVaultError` exception family. Several exception
subclasses also derive from a familiar Python builtin (``KeyError``,
``MemoryError``, ``OSError``, ``FileNotFoundError``, ``ValueError``) so callers
can catch either.
"""

from ._vectorvault import (  # noqa: F401
    Engine,
    Collection,
    CollectionInfo,
    DistanceMetric,
    IndexType,
    version,
    VectorVaultError,
    NameConflictError,
    InvalidNameError,
    InvalidDimensionalityError,
    InvalidMetricError,
    DeletionFailedError,
    NotFoundError,
    DimensionMismatchError,
    InvalidValueError,
    InvalidIdentifierError,
    InvalidParameterError,
    InvalidArgumentError,
    UndefinedDistanceError,
    AllocationFailureError,
    WriteFailureError,
    SnapshotNotFoundError,
    UnsupportedVersionError,
    CorruptionError,
)

try:
    from ._vectorvault import __version__  # noqa: F401
except ImportError:  # pragma: no cover - defensive fallback
    __version__ = version()

__all__ = [
    "Engine",
    "Collection",
    "CollectionInfo",
    "DistanceMetric",
    "IndexType",
    "version",
    "__version__",
    "VectorVaultError",
    "NameConflictError",
    "InvalidNameError",
    "InvalidDimensionalityError",
    "InvalidMetricError",
    "DeletionFailedError",
    "NotFoundError",
    "DimensionMismatchError",
    "InvalidValueError",
    "InvalidIdentifierError",
    "InvalidParameterError",
    "InvalidArgumentError",
    "UndefinedDistanceError",
    "AllocationFailureError",
    "WriteFailureError",
    "SnapshotNotFoundError",
    "UnsupportedVersionError",
    "CorruptionError",
]
