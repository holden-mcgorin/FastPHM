from rulframework.data.feature.KurtosisFeatureExtractor import KurtosisFeatureExtractor
from rulframework.data.feature.RMSFeatureExtractor import RMSFeatureExtractor
from rulframework.data.raw.PHM2012DataLoader import PHM2012DataLoader
from rulframework.data.raw.XJTUDataLoader import XJTUDataLoader
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.util.Plotter import Plotter
from rulframework.util.Timer import Timer

if __name__ == '__main__':
    data_loader = XJTUDataLoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    # data_loader = PHM2012DataLoader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')
    feature_extractor = RMSFeatureExtractor(data_loader.continuum)
    # feature_extractor = KurtosisFeatureExtractor(data_loader.continuum)
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)

    for bearing_name in data_loader:
        Timer.start()
        bearing = data_loader.get_bearing(bearing_name, columns='Horizontal Vibration')
        bearing.feature_data = feature_extractor.extract(bearing.raw_data)
        stage_calculator.calculate_state(bearing)
        Plotter.feature(bearing, is_staged=True, is_save=False)
        print(bearing)
        Timer.stop()
