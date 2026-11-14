"""Domain errors. Each carries the HTTP status it should map to, so routes stay thin."""


class AppError(Exception):
    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(AppError):
    status_code = 415


class FileTooLargeError(AppError):
    status_code = 413


class EmptyDocumentError(AppError):
    status_code = 422


class ExternalServiceError(AppError):
    """Redis / Qdrant / LLM provider is unreachable or misbehaving."""

    status_code = 503
