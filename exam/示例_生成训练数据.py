from core.data_manager.feature_data.RMSFeatureExtractor import RMSFeatureExtractor
from core.data_manager.raw_data.XJTUDataLoader import XJTUDataLoader
from core.data_manager.train_data.SlideWindowDataGenerator import SlideWindowDataGenerator
from core.entity.Bearing import Bearing
from core.stage_calculator.BearingStageCalculator import BearingStageCalculator
from core.stage_calculator.eol_calculator.NinetyFivePercentRMSEoLCalculator import NinetyFivePercentRMSEoLCalculator
from core.stage_calculator.eol_calculator.TenMaxAmplitudeEoLCalculator import TenMaxAmplitudeEoLCalculator
from core.stage_calculator.eol_calculator.EightMeanRMSEoLCalculator import EightMeanRMSEoLCalculator
from core.stage_calculator.fpt_calculator.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

if __name__ == '__main__':
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    feature_extractor = RMSFeatureExtractor(32768)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyFivePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator)

    bearing = data_loader.get_bearing("Bearing1_3", column='Horizontal Vibration')
    bearing.feature_data = feature_extractor.extract(bearing.raw_data)
    stage_calculator.calculate_state(bearing, 32768)
    bearing.plot_feature()

    data_generator = SlideWindowDataGenerator(10)
    bearing.train_data = data_generator.generate_data(bearing.feature_data)
    print(bearing.train_data)
