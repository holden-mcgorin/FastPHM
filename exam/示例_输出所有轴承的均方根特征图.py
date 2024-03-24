from rulframework.data_manager.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data_manager.raw.PHM2012DataLoader import PHM2012DataLoader
from rulframework.data_manager.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.stage_calculator.BearingStageCalculator import BearingStageCalculator
from rulframework.stage_calculator.eol.NinetyFivePercentRMSEoLCalculator import NinetyFivePercentRMSEoLCalculator
from rulframework.stage_calculator.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

if __name__ == '__main__':
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    # data_loader = PHM2012DataLoader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')
    feature_extractor = RMSFeatureExtractor(32768)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyFivePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.span)

    for bearing_name in data_loader.all:
        bearing = data_loader.get_bearing(bearing_name, columns='Horizontal Vibration')
        bearing.feature_data = feature_extractor.extract(bearing.raw_data)
        stage_calculator.calculate_state(bearing)
        bearing.plot_feature()
