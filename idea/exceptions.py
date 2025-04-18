class IDEAError(Exception):
    """Base class for other exceptions."""

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message
