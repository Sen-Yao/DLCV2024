from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 生成 Half Moon 数据集
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 可视化数据集
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='orange', label='Class 1')
plt.title('Half Moon Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

