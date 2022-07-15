import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep


# 目标函数
def f(x):
    return (np.sin(x) * x + 0.5 * x)


# 三次样条插值的关键是splrep和splev
xx = np.linspace(0, 2 * np.pi, 50)
# x是插值节点,替换成原有的插值
x = np.array([0,1.1,1.3,1.5,1.6,1.9,2,3.5,4.1,5.3,6.5])
ipo = splrep(x, f(x), k=3)  # k == 3表示三次样条插值
# iy是其他节点的插值效果，替换成需要生成的插值
iy = splev(xx, ipo)

# 可视化效果
plt.plot(xx, f(xx), 'b', label='f(x)')
plt.plot(xx, iy, 'r', label='interpolation')
plt.legend(loc=0)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()