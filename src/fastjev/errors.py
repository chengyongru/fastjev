"""Public exception hierarchy for the fastjev SDK."""


class FastJevError(Exception):
    """Base class for SDK errors."""


class ValidationError(FastJevError):
    """A decision request is invalid for the public SDK contract."""


class InputTooLongError(ValidationError):
    """A rendered decision exceeds the configured backend input limit."""


class ModelLoadError(FastJevError):
    """A backend could not load its configured model."""


class BackendProtocolError(FastJevError):
    """A backend returned data that violates the scoring protocol."""
