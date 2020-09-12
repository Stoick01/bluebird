class ProgressBar():
    def __init__(self, total: int, num_epochs: int):
        self.total = total
        self.num_epochs = num_epochs
        self.length = len(str(num_epochs))

    def print_bar(self, iteration: int, epoch: int, loss: int):
        percent = ("{0:.2f}").format(100 * iteration / float(self.total))

        filled_length = int(50 * iteration // self.total)
        if filled_length == 50:
            bar = '=' * 49 + ">"
        else:
            bar = '=' * filled_length + '-' * (50 - filled_length)

        print(f'Epoch: {epoch:{self.length}}/{self.num_epochs} |{bar}| {percent}% loss: {loss:.4f}', end="\r")
        if iteration == self.total:
            print()
