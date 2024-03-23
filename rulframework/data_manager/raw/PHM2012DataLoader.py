from typing import Dict

from rulframework.data_manager.raw.ABCDataLoader import ABCDataLoader


class PHM2012DataLoader(ABCDataLoader):

    def _build_item_dict(self, root) -> Dict[str, str]:
        pass

    def load(self, item_name) -> object:
        pass
