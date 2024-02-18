from core.data_manager.feature_data.RMSFeatureExtractor import RMSFeatureExtractor
from core.data_manager.raw_data.XJTUDataLoader import XJTUDataLoader
from core.entity.Bearing import Bearing
from core.stage_calculator.BearingStageCalculator import BearingStageCalculator
from core.stage_calculator.eol_calculator.TenAmplitudeEoLCalculator import TenAmplitudeEoLCalculator
from core.stage_calculator.fpt_calculator.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator

if __name__ == '__main__':
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    bearing = data_loader.get_bearing("Bearing1_3", column='Horizontal Vibration')
    bearing.feature_data = RMSFeatureExtractor(32768).extract(bearing.raw_data)
    BearingStageCalculator(ThreeSigmaFPTCalculator(), TenAmplitudeEoLCalculator()).calculate_state(bearing, 32768)
    bearing.plot_feature()

    # print(bearing.stage_data)
