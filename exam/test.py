import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制曲线
plt.plot(y1, x, label='sin(x)')
plt.plot(y2, x, label='cos(x)')

# 使用 plt.fill_between 沿着 y 轴绘制填充区域
plt.fill_betweenx(x, y1, y2, where=(y1 >= y2), color='lightgray', alpha=0.3)

# 添加图例和标签
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Example of plt.fill_between')

# 显示图形
plt.show()
