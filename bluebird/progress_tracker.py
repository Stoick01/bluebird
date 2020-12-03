"""
Progress tracker
================

Simple progress bar that shows progress through each epoch.
"""

class ProgressBar():
    """
    Simple progress bar, used for visual representation durign training.

    Example::

        pb = ProgressBar(5000, 20)
        pb.print_bar(500, 2, 0.976)

    Output::

        Epoch:  1/20 |===========...................| 25.5% loss: 0.976

    """

    def __init__(self, total: int, num_epochs: int) -> None:
        """
        Initalizes the object.
        
        Args:
            total (int): total number of iteration per epoch
            num_epochs (int): total number of epochs

        """
        self.total = total
        self.num_epochs = num_epochs
        self.length = len(str(num_epochs))

    def print_bar(self, iteration: int, epoch: int, loss: float) -> None:
        """
        Used to print each iteration of progress bar

        Args:
            iteration (int): current iteration
            epoch (int): current epoch
            loss (float): current loss

        Returns:
            Nothing
            
        """

        percent = ("{0:.2f}").format(100 * iteration / float(self.total))

        filled_length = int(50 * iteration // self.total)
        if filled_length == 50:
            bar = '=' * 49 + ">"
        else:
            bar = '=' * filled_length + '-' * (50 - filled_length)

        print(f'Epoch: {epoch:{self.length}}/{self.num_epochs} |{bar}| {percent:>6}% loss: {loss:.4f}', end="\r")
        if filled_length == 50:
            print()
