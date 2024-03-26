from rulframework.data.raw.PHM2012DataLoader import PHM2012DataLoader
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.util.Timer import Timer

if __name__ == '__main__':
    timer = Timer()

    timer.start()
    data_loader_phm = PHM2012DataLoader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')
    bearing1_1 = data_loader_phm.get_bearing('Bearing1_1')
    bearing1_1.plot_raw()
    timer.stop()

    timer.start()
    data_loader_xj = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    bearing3_1 = data_loader_xj.get_bearing('Bearing2_1', columns='Horizontal Vibration')
    bearing3_1.plot_raw()
    timer.stop()
