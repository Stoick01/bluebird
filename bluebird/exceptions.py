"""
Custom exceptions
"""

class TypeException(TypeError):
    def __init__(self, *args):
        if len(args) == 2:
            self.message = f"{args[0]} should be a {args[1]}"
        else:
            self.message = None

    def __str__(self):
        return f"TypeError: {self.message}."

        