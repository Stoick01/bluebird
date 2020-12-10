"""
Exceptions
==========

Custom exceptions used for better handling of errors.

Designed to help the user find the error faster.
"""

class TypeException(TypeError):
    """
    Handels the type missmatch, prints variable and the type it should be.

    Example::

        raise TypeException("input", "Tensor")
    """
    
    def __init__(self, *args):
        """
        Initializes the object.

        Args:
            args[0] (:obj:`str`): name of variable
            args[1] (:obj:`str`): expected type
        """

        if len(args) == 2:
            self.message = f"{args[0]} should be a {args[1]}"
        else:
            self.message = None

    def __str__(self):
        return f"TypeError: {self.message}."

        