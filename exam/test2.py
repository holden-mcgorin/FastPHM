import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制曲线
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')

# 使用 plt.fill_between 填充两条曲线之间的颜色
plt.fill_between(x, y1, y2, color='lightgray', alpha=0.3)

# 添加图例和标签
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Example of plt.fill_between')

# 显示图形
plt.show()
