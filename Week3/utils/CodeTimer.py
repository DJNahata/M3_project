import time

class CodeTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time() - self.start
        print("{}: {} s".format(self.name, self.end))
