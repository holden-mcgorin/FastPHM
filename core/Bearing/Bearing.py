class Bearing:
    """
    轴承对象
    """

    def __init__(self, name, raw_data=None, feature_data=None, train_data=None, stage_data=None):
        self.name = name
        self.raw_data = raw_data
        self.feature_data = feature_data
        self.train_data = train_data
        self.stage_data = stage_data
