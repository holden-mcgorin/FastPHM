from rulframework.data_manager.feature_data.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data_manager.raw_data.XJTUDataLoader import XJTUDataLoader
from rulframework.stage_calculator.BearingStageCalculator import BearingStageCalculator
from rulframework.stage_calculator.eol_calculator.NinetyFivePercentRMSEoLCalculator import NinetyFivePercentRMSEoLCalculator
from rulframework.stage_calculator.fpt_calculator.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

if __name__ == '__main__':
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = RMSFeatureExtractor(32768)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyFivePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, 32768)

    for bearing_name in data_loader.get_bearings_name():
        bearing = data_loader.get_bearing(bearing_name, column='Horizontal Vibration')
        bearing.feature_data = feature_extractor.extract(bearing.raw_data)
        stage_calculator.calculate_state(bearing)
        bearing.plot_feature()
