from sacred import Experiment
from sacred.observers import FileStorageObserver


def init_exp():
    global ex
    ex = Experiment("centerpoint_train")
    ex.observers.append(FileStorageObserver("my_runs"))
