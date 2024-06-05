import numpy as np


def kernel_matrix(X, Y, kernel_func):
    """
    计算核矩阵
    Args:
        X: 第一个数组，shape为 (n_samples_X, n_features)
        Y: 第二个数组，shape为 (n_samples_Y, n_features)
        kernel_func: 核函数

    Returns:
        K: 核矩阵，shape为 (n_samples_X, n_samples_Y)
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    K = np.zeros((n_samples_X, n_samples_Y))
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            K[i, j] = kernel_func(X[i], Y[j])
    return K


def gaussian_kernel(x, y, sigma=1.0):
    """
    高斯核函数
    """
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def mmd(X, Y, kernel_func):
    """
    计算 MMD
    Args:
        X: 第一个数组，shape为 (n_samples_X, n_features)
        Y: 第二个数组，shape为 (n_samples_Y, n_features)
        kernel_func: 核函数

    Returns:
        mmd: MMD 值
    """
    K_XX = kernel_matrix(X, X, kernel_func)
    K_YY = kernel_matrix(Y, Y, kernel_func)
    K_XY = kernel_matrix(X, Y, kernel_func)
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    return mmd


# 示例用法
# X = np.random.randn(100, 2)  # 第一个数组
# Y = np.random.randn(100, 2)  # 第二个数组
X = np.random.normal(100, 10, 100)
Y = np.random.normal(100, 10, 500)
print(f'x = {X}')
print(f'y = {Y}')
mmd_value = mmd(X, Y, gaussian_kernel)
print("MMD:", mmd_value)
