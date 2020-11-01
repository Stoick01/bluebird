class ProgressBar():
    """
    Simple progress bar, used for visual representation durign training

    Args:
        total: total number of iteration per epoch, Type: int
        num_epochs: total number of epochs, Type: int

    Example:
        >>> pb = ProgressBar(5000, 20)

    """

    def __init__(self, total: int, num_epochs: int):
        self.total = total
        self.num_epochs = num_epochs
        self.length = len(str(num_epochs))

    def print_bar(self, iteration: int, epoch: int, loss: float):
        """
        Used to print each iteration of progress bar

        Args:
            iteration: current iteration, Type: int
            epoch: current epoch, Type: int
            loss: current loss, Type: float

        Example:
            >>> pb = ProgressBar(5000, 20)
            >>> pb.print_bar(500, 2, 0.976)

        Output Example:
            Epoch:  1/20 |===========...................| 25.5% loss: 0.976

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
