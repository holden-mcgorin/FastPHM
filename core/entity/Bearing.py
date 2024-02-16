from matplotlib import pyplot as plt


class Bearing:
    """
    轴承对象
    """

    # 常量，生成的图片大小
    FIG_SIZE = (10, 6)

    def __str__(self) -> str:
        return self.name

    def __init__(self, name, raw_data=None, feature_data=None, train_data=None, stage_data=None, raw_data_loc=None):
        self.name = name
        self.raw_data = raw_data
        self.feature_data = feature_data
        self.train_data = train_data
        self.stage_data = stage_data
        self.raw_data_loc = raw_data_loc

    def plot_raw_data(self, single_signal=None, is_save=False):
        """
        绘画原始振动信号图像
        :param single_signal: 字符串，列名，默认将水平信号与垂直信号都画出，可用此变量指定只画一个方向信号
        :param is_save: 是否保存图片，默认不保存
        :return:
        """
        if self.raw_data is None:
            raise Exception("此轴承原始振动信号变量raw_data为None，请先使用数据加载器加载原始数据赋值给此轴承对象！")

        plt.figure(figsize=self.FIG_SIZE)

        if single_signal is not None:
            plt.plot(self.raw_data[single_signal], label=single_signal)
        else:
            plt.plot(self.raw_data['Horizontal Vibration'], label='Horizontal Vibration')
            plt.plot(self.raw_data['Vertical Vibration'], label='Vertical Vibration')
        plt.title(self.name + ' Vibration Signals')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('vibration')
        plt.legend()
        if is_save:
            plt.savefig(self.name + '.png', dpi=300)
        plt.show()

    def plot_feature_data(self):
        plt.figure(figsize=self.FIG_SIZE)

        for key in self.feature_data:
            plt.plot(self.feature_data[key], label=key)

        plt.title(self.name + ' Vibration Signals')
        plt.xlabel('Time (Sample Index)')
        plt.ylabel('vibration')
        plt.legend()
        plt.show()

