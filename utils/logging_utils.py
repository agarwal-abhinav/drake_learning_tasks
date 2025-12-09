class IterTee:
    def __init__(self, stdout, global_file):
        self.stdout = stdout
        self.global_file = global_file
        self.iter_file = None

    def set_iter_file(self, f):
        self.iter_file = f

    def write(self, data):
        self.stdout.write(data)
        if self.global_file is not None and not self.global_file.closed:
            self.global_file.write(data)
        if self.iter_file is not None and not self.iter_file.closed:
            self.iter_file.write(data)

    def flush(self):
        self.stdout.flush()
        if self.global_file is not None and not self.global_file.closed:
            self.global_file.flush()
        if self.iter_file is not None and not self.iter_file.closed:
            self.iter_file.flush()

