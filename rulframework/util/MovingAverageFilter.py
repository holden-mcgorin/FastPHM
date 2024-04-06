class MovingAverageFilter:
    """
    移动平均滤波器
    1. 可用来平滑滚动预测产生的断层
    """
    def __init__(self, window_size) -> None:
        self.window_size = window_size

    def moving_average(self, *args):
        """
        计算移动平均滤波器
        参数：
        data: 待滤波的数据，可以是一个列表或 NumPy 数组
        window_size: 滑动窗口的大小，用于计算平均值
        返回值：
        filtered_data: 滤波后的数据，与原始数据长度相同
        """
        result = []
        for arg in args:
            # 检查窗口大小是否有效
            if self.window_size < 1 or self.window_size > len(arg):
                raise ValueError("Invalid window size")

            filtered_data = []
            # 遍历数据，对每个数据点应用移动平均滤波
            for i in range(len(arg)):
                # 计算窗口的开始和结束索引
                start_index = max(0, i - self.window_size + 1)
                end_index = i + 1

                # 获取窗口内的数据
                window_data = arg[start_index:end_index]

                # 计算窗口内数据的平均值
                average = sum(window_data) / len(window_data)

                # 将平均值添加到滤波后的数据列表中
                filtered_data.append(average)

            result.append(filtered_data)

        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)
