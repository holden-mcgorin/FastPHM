from rulframework.data.FeatureExtractor import FeatureExtractor
from rulframework.data.processor.RMSProcessor import RMSProcessor
from rulframework.data.loader.bearing.PHM2012Loader import PHM2012Loader
from rulframework.data.stage.BearingStageCalculator import BearingStageCalculator
from rulframework.data.stage.eol.NinetyThreePercentRMSEoLCalculator import NinetyThreePercentRMSEoLCalculator
from rulframework.data.stage.fpt.ThreeSigmaFPTCalculator import ThreeSigmaFPTCalculator
from rulframework.system.Logger import Logger
from rulframework.util.Plotter import Plotter

if __name__ == '__main__':
    # data_loader = XJTULoader('D:\\data\\dataset\\XJTU-SY_Bearing_Datasets')
    data_loader = PHM2012Loader('D:\\data\\dataset\\phm-ieee-2012-data-challenge-dataset-master')
    feature_extractor = FeatureExtractor(RMSProcessor(data_loader.continuum))
    # feature_extractor = FeatureExtractor(KurtosisProcessor(data_loader.continuum))
    fpt_calculator = ThreeSigmaFPTCalculator()
    eol_calculator = NinetyThreePercentRMSEoLCalculator()
    stage_calculator = BearingStageCalculator(fpt_calculator, eol_calculator, data_loader.continuum)

    for bearing_name in data_loader:
        bearing = data_loader(bearing_name, columns='Horizontal Vibration')
        # bearing = data_loader(bearing_name)
        feature_extractor(bearing)
        # stage_calculator.calculate_state(bearing)
        Plotter.feature(bearing, is_staged=False)
        Logger.info(str(bearing))
