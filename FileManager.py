import os


class FileManager:

    def __init__(self, parameters, train_progress_path, validation_progress_path, base_path, base_name):
        if not base_name.endswith("/"):
            base_name += "/"
        if not base_path.endswith("/"):
            base_path += "/"
        if not train_progress_path.endswith("/"):
            train_progress_path += "/"
        if not validation_progress_path.endswith("/"):
            validation_progress_path += "/"

        name = base_name[:-1]
        base_path = base_path + base_name
        self.parameters = parameters
        self.train_progress_path = base_path + train_progress_path
        self.validation_progress_path = base_path + validation_progress_path
        self.base_path = base_path
        self.dna_path = self.base_path + name + "_dna.txt"

        self.create_paths()

    def create_paths(self):
        if not os.path.isdir(self.base_path):
            os.makedirs(self.base_path)
        if not os.path.isdir(self.train_progress_path):
            os.makedirs(self.train_progress_path)
        if not os.path.isdir(self.validation_progress_path):
            os.makedirs(self.validation_progress_path)
        if not os.path.isdir(self.base_path + "saved_graphs"):
            os.makedirs(self.base_path + "saved_graphs")

    def network_save_path(self, step):
        return self.base_path + "saved_graphs/step_%d/model.ckpt" % step
