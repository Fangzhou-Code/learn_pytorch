import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
a = 0.01

def forward(x):
    return w * x



def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_predict = forward(x)
        cost += (y_predict-y)**2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2*x*(x*w-y)
    return grad / len(xs)

print('predict:(before trainning):', 4, forward(4))
cost_list = []
epoch_list = []
for epoch in range(100):
    epoch_list.append(epoch)
    cost_val = cost(x_data,y_data)
    cost_list.append(cost_val)
    w -= a*gradient(x_data, y_data)
    print("Epoch=",epoch,"w=",w,"loss=",cost_val)
print('predict:(after trainning):', 4, forward(4))

plt.plot(epoch_list,cost_list)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()