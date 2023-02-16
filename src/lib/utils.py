import pickle


def save_f(filename: str, obj: any) -> None:
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_f(filename: str) -> any:
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
