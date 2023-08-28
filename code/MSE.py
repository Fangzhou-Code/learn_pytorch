import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.5, 4.5, 6.5]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

w_list = np.arange(0.0, 4.0, 0.1)
b_list = np.arange(0.0, 4.0, 0.1)
mse_list = []

for w in w_list:
    sub_list = []
    for b in b_list:
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val, w, b)
            l_sum += loss_val
        sub_list.append(l_sum / len(x_data))
    mse_list.append(sub_list)

# 转换为网格数据
X, Y = np.meshgrid(w_list, b_list)
Z = np.array(mse_list)

# 创建图形对象和子图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面图
ax.plot_surface(X, Y, Z)

# 设置坐标轴标签
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('MSE')

# 显示图形
plt.show()
