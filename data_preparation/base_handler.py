
import datasets

class BaseHandler:

    url = None
    subset = None
    split = None

    kind = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    

    def load_dataset(self):
        if self.subset is not None:
            return datasets.load_dataset(self.url, self.subset, split=self.split)
        return datasets.load_dataset(self.url, split=self.split)


    def full_map(self, example):
        inp, out, form = self.map(example)
        return {
            "source": f"{self.url}/{self.subset}" if self.subset is not None else self.url,
            "kind": self.kind,
            "format": form,
            "input": inp,
            "output": out,
        }


    def map(self, example):
        raise NotImplementedError("Subclasses must implement this method.")
    

    def filter(self, example):
        return True
    