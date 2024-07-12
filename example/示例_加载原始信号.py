from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.PHM2012DataLoader import PHM2012DataLoader
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.util.Plotter import Plotter
from rulframework.util.Timer import Timer

if __name__ == '__main__':
    Timer.start()
    data_loader_phm = PHM2012DataLoader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')
    bearing1_1 = data_loader_phm.get_bearing('Bearing1_1')
    Plotter.raw(bearing1_1)
    feature_extractor = RMSFeatureExtractor(data_loader_phm.continuum)
    bearing1_1.feature_data = feature_extractor.extract(bearing1_1.raw_data)
    Plotter.feature(bearing1_1)
    Timer.stop()

    # Timer.start()
    # data_loader_xj = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    # bearing3_1 = data_loader_xj.get_bearing('Bearing2_1', columns='Horizontal Vibration')
    # Plotter.raw(bearing3_1)
    # Timer.stop()
