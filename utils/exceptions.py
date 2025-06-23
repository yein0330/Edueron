# src/utils/exceptions.py

class QueryDocException(Exception):
    """Base exception for QueryDoc application."""
    pass

class PDFExtractionError(QueryDocException):
    """Raised when PDF extraction fails."""
    pass

class EmbeddingError(QueryDocException):
    """Raised when embedding generation fails."""
    pass

class SearchError(QueryDocException):
    """Raised when search operation fails."""
    pass

class AuthenticationError(QueryDocException):
    """Raised when authentication fails."""
    pass

class FileNotFoundError(QueryDocException):
    """Raised when required file is not found."""
    pass

class ModelLoadError(QueryDocException):
    """Raised when model loading fails."""
    pass