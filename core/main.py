from core.data_manager.feature_data.RMSFeatureGenerator import RMSFeatureGenerator
from core.data_manager.raw_data.XJTUDataLoader import XJTUDataLoader
from core.entity.Bearing import Bearing

if __name__ == '__main__':
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    bearings = list(data_loader.get_all_bearings_name())
    generator = RMSFeatureGenerator(32768)
    for b in bearings:
        bearing = Bearing(b)
        print(bearing)
        bearing.raw_data = data_loader.load_raw_data(bearing.name)
        bearing.feature_data = generator.extract_feature(bearing.raw_data)
        bearing.plot_feature_data()
    # print(ThreeSigmaFPTCalculator().get_fpt(bearing.feature_data['Horizontal Vibration']))
