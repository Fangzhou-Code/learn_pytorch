'''
batch 指的是全体，这里batch实际指的是mini—batch
'''
import matplotlib.pyplot as plt
import numpy as np

x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])

w = 1.0
a = 0.01
batch_size = 2  # Mini-batch大小


def forward(x):
    return w * x


def cost(y_predict, y_true):
    return np.mean((y_predict - y_true) ** 2)


def gradient(xs, ys):
    return np.mean(2 * xs * (xs * w - ys))


print('Predict (before training):', forward(4))
cost_list = []
epoch_list = []

num_epochs = 100
num_batches = len(x_data) // batch_size

for epoch in range(num_epochs):
    epoch_list.append(epoch)
    epoch_cost = 0.0

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        x_batch = x_data[start_idx:end_idx]
        y_batch = y_data[start_idx:end_idx]

        y_predict = forward(x_batch)
        batch_loss = cost(y_predict, y_batch)
        epoch_cost += batch_loss

        w -= a * gradient(x_batch, y_batch)

    cost_list.append(epoch_cost / num_batches)
    print("Epoch =", epoch, "w =", w, "loss =", epoch_cost / num_batches)

print('Predict (after training):', forward(4))

plt.plot(epoch_list, cost_list)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()
