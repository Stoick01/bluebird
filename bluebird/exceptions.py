"""
Custom exceptions
"""

class TypeException(TypeError):
    """
    Simple type exception

    Args:
        args[0]: name of variable, Type: String
        args[1]: type that variable should be, Type: String

    Example:
        >>> raise TypeException("input", "Tensor")
    """
    def __init__(self, *args):
        if len(args) == 2:
            self.message = f"{args[0]} should be a {args[1]}"
        else:
            self.message = None

    def __str__(self):
        return f"TypeError: {self.message}."

        