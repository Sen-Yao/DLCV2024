import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 激活函数及其导数
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.power(np.tanh(x), 2)

# 前向传播
def forward_propagation(X, model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return a1, probs

# 计算损失
def compute_loss(probs, y, model, reg_lambda):
    num_examples = probs.shape[0]
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = 0.5 * reg_lambda * (np.sum(np.square(model['W1'])) + np.sum(np.square(model['W2'])))
    return data_loss + reg_loss

# 反向传播
def backward_propagation(X, y, a1, probs, model, reg_lambda):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    num_examples = X.shape[0]

    delta3 = probs
    delta3[range(num_examples), y] -= 1
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)

    delta2 = delta3.dot(W2.T) * tanh_derivative(a1)
    dW1 = X.T.dot(delta2)
    db1 = np.sum(delta2, axis=0)

    # 添加正则化项
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1

    return dW1, db1, dW2, db2

# 更新参数
def update_parameters(model, dW1, db1, dW2, db2, learning_rate):
    model['W1'] -= learning_rate * dW1
    model['b1'] -= learning_rate * db1
    model['W2'] -= learning_rate * dW2
    model['b2'] -= learning_rate * db2

# 训练模型
def train(X, y, hidden_size=20, num_passes=100000, learning_rate=0.001, reg_lambda=0.001):
    input_size = X.shape[1]
    output_size = len(np.unique(y))

    # 初始化参数
    model = {
        'W1': np.random.randn(input_size, hidden_size) * 0.01,
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size) * 0.01,
        'b2': np.zeros((1, output_size))
    }

    for i in range(num_passes):
        a1, probs = forward_propagation(X, model)
        loss = compute_loss(probs, y, model, reg_lambda)
        dW1, db1, dW2, db2 = backward_propagation(X, y, a1, probs, model, reg_lambda)
        update_parameters(model, dW1, db1, dW2, db2, learning_rate)

        # 每1000次迭代输出一次损失
        if i % 500 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")

    return model

# 预测
def predict(model, X):
    _, probs = forward_propagation(X, model)
    return np.argmax(probs, axis=1)

# 可视化结果
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# 主函数
if __name__ == "__main__":
    # 生成 Half Moon 数据集
    X, y = make_moons(n_samples=1000, noise=0.1)
    y = y.reshape(-1)

    # 训练模型
    model = train(X, y)

    # 可视化决策边界
    plot_decision_boundary(model, X, y)
