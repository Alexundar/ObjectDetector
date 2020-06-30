from abc import ABC, abstractmethod


class AbstractCfgParser(ABC):
    @abstractmethod
    def parse_cfg(self, cfgfile):
        pass