from abc import ABC, abstractmethod
from typing import Iterable


AFTER = "after"
BEFORE = "before"


def mechanism(method: callable):
    method_name = method.__name__

    def mechanism_handled(self, *args, **kwargs):
        if method_name in self.mechanism[BEFORE]:
            self.mechanism[BEFORE][method_name](self)
        returned_value = method(self, *args, **kwargs)
        if method_name in self.mechanism[AFTER]:
            returned_value = self.mechanism[AFTER][method_name](self, returned_value)
        return returned_value

    return mechanism_handled


class AbstractMechanism(ABC):
    def __init__(self):
        self.registered = []
        self.main()

    @abstractmethod
    def main(self):
        pass

    def init(self, population_object):
        pass

    def register(self, method_name: str, on: str, mechanism_func: callable):
        self.registered.append({
            "method_name": method_name,
            "on": on,
            "mechanism_func": mechanism_func
        })

    def __call__(self, *args, **kwargs) -> Iterable[dict]:
        return self.registered


# For use in classes
def mechanism_init(self):
    self.mechanism = {BEFORE: {}, AFTER: {}}


def register_mechanism_function(self, method_name: str, on: str, mechanism_func: callable):
    if on not in [BEFORE, AFTER]:
        raise Exception("\"on\" parameter should be one of \"before\" or \"after\"")
    self.mechanism[on][method_name] = mechanism_func


def register_mechanism_object(self, mechanism_instance: AbstractMechanism):
    mechanism_instance.init(self)
    mechanisms = mechanism_instance()
    for mechanism_row in mechanisms:
        self.register_mechanism_function(
            mechanism_row["method_name"],
            mechanism_row["on"],
            mechanism_row["mechanism_func"]
        )
