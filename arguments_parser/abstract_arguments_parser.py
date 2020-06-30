from abc import ABC, abstractmethod


class AbstractArgumentsParser(ABC):
    @abstractmethod
    def parse_arguments(self):
        pass
