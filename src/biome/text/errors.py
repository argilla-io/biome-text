from fastapi import HTTPException


class BaseError(Exception):
    """Base error. This class could include common error attributes or methods"""

    pass


class ValidationError(BaseError):
    """Base error for data validation"""

    pass


class MissingArgumentError(ValidationError):
    """Error related with input params"""

    def __init__(self, arg_name: str):
        super(MissingArgumentError, self).__init__()
        self.arg_name = arg_name

    def __str__(self) -> str:
        return f"Missing argument '{self.arg_name}'"


class ActionNotSupportedError(ValidationError):
    """Raised when an action is not supported for a given component state"""


class EmptyVocabError(ValidationError):
    """Error related with using empty vocabs for a training"""

    pass


class WrongValueError(ValidationError):
    """Wrong value error"""


class http_error_handling:
    """Error handling for http error transcription"""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, ValidationError):
            raise HTTPException(status_code=400, detail=str(exc_val))
        if isinstance(exc_val, Exception):
            # Common http error handling
            raise HTTPException(status_code=500, detail=str(exc_val))
