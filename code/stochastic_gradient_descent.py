import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 # 初始权重
a = 0.01 # 学习率

def forward(x):
    return w * x



# 定义在单个样本上叫loss function，定义在整个训练集上叫cost function
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    return 2 * x * (forward(x)-y)

print('predict:(before trainning):', 4, forward(4))

# for epoch in range(100):
#     for x,y in zip(x_data, y_data):
#         grad = gradient(x, y)
#         w -= a * grad
#         print("\tgrad:", x, y, grad)
#         l = loss(x, y)
#     print("progress:",epoch,"w=", w, "loss=", l, "x=", x, "y=", y)

'''
会发现一些有趣的现象：
1. 尽管在一次迭代中w在不断更新，但不意味着在一个数据集内不同的点对应的loss值会降低。
因为初始的时候不同的点loss值就不同
2. w 不停迭代会无限接近于我们想要的值但不会等于
3. 随机梯度下降效果可能更好，但是不能并行，时间复杂度更高。因为下次的预测值取决于上一次的w更新值。只能依次运行
'''
for epoch in range(100):
    print("progress:", epoch)
    for x,y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= a * grad
        l = loss(x, y)
        print("\tgrad:", x, y, grad, "w=", w,"\tloss=",l)

print('predict:(after trainning):', 4, forward(4))
