"""
Author: David Torpey

License: Apache 2.0

Module defining all Theama base exceptions.
"""


class TheamaException(Exception):
    """
    Base class for Theama exceptions.
    """

    def __init__(self, *args, **kwargs):
        default_message = 'Theama exception was thrown.'

        if not (args or kwargs):
            args = (default_message,)

        # Call super constructor
        super().__init__(*args, **kwargs)
