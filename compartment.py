from typing import Dict

class Compartment(object):

    values: Dict[float, float]

    def __init__(self, name: str, initial_value: float = 0.0):
        self.name = name
        self.values = dict()
        self.values[0] = initial_value
