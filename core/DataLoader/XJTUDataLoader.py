import os

from core.DataLoader.DataLoader import DataLoader


class XJTUDataLoader(DataLoader):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        for condition in ['35Hz12kN', '37.5Hz11kN', '40Hz10kN']:
            condition_dir = os.path.join(root_dir, condition)
            for bearing_name in os.listdir(condition_dir):
                self.bearing_dict[bearing_name] = os.path.join(root_dir, condition, bearing_name)
        for key, value in self.bearing_dict.items():
            print(f"轴承: {key}，位置: {value}")

    def load_bearing(self, bearing_name):
        pass


if __name__ == '__main__':
    data_loader = XJTUDataLoader('D://data//dataset//XJTU-SY_Bearing_Datasets')
