"""Public Python SDK for backend-neutral semantic decisions."""

__version__ = "0.2.0"

from .backends import (
    BackendCapabilities,
    BackendInfo,
    BackendOption,
    BackendRequest,
    BackendResult,
    ExLlamaV3Backend,
    LlamaCppBackend,
    MLXBackend,
    ScoringBackend,
    TorchBackend,
    VLLMBackend,
)
from .client import FastJev
from .calibration import CalibrationSample, TemperatureCalibration
from .compat import SystemOneAdapter
from .errors import (
    BackendProtocolError,
    BackendExecutionError,
    FastJevError,
    InputTooLongError,
    ModelLoadError,
    ValidationError,
)
from .types import (
    Boolean,
    Calibration,
    Choice,
    Decision,
    Level,
    Option,
    Provenance,
    Score,
    Timing,
    Uncertainty,
    Usage,
)

# Retained low-level imports for compatibility with the first fastjev release.
from ._runtime.core import load_causal_model
from ._runtime.direct import score as score_direct
from .compat.wire import (
    QuestionSpec,
    SystemOneService,
    SystemOneValidationError,
    distribution_confidence,
    request_rows,
    response_from_results,
)

__all__ = [
    "BackendCapabilities",
    "BackendExecutionError",
    "BackendInfo",
    "BackendOption",
    "BackendProtocolError",
    "BackendRequest",
    "BackendResult",
    "Boolean",
    "Calibration",
    "CalibrationSample",
    "Choice",
    "Decision",
    "ExLlamaV3Backend",
    "FastJev",
    "FastJevError",
    "InputTooLongError",
    "Level",
    "LlamaCppBackend",
    "MLXBackend",
    "ModelLoadError",
    "Option",
    "Provenance",
    "QuestionSpec",
    "Score",
    "ScoringBackend",
    "SystemOneService",
    "SystemOneAdapter",
    "SystemOneValidationError",
    "Timing",
    "TemperatureCalibration",
    "TorchBackend",
    "Uncertainty",
    "Usage",
    "ValidationError",
    "VLLMBackend",
    "distribution_confidence",
    "load_causal_model",
    "request_rows",
    "response_from_results",
    "score_direct",
    "__version__",
]
