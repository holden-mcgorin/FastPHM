from rulframework.data_manager.raw.PHM2012DataLoader import PHM2012DataLoader
from rulframework.data_manager.raw.XJTUDataLoader import XJTUDataLoader

if __name__ == '__main__':
    data_loader_phm = PHM2012DataLoader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')
    data_loader_xj = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')

    bearing1_1 = data_loader_xj.get_bearing('Bearing1_1', columns='Horizontal Vibration')
    bearing1_1.plot_raw()

    bearing1_6 = data_loader_phm.get_bearing('Bearing1_6', columns='Vertical Vibration')
    bearing1_6.plot_raw()
