import scipy.io as sio
import numpy as np
import pandas as pd


def test_libraries():
    print("开始测试库的可用性...")
    print("=" * 50)

    # 测试numpy
    try:
        # 创建一个numpy数组
        arr = np.array([1, 2, 3, 4, 5])
        print("✓ NumPy 可用")
        print(f"  示例数组: {arr}")
        print(f"  数组形状: {arr.shape}")
    except Exception as e:
        print(f"✗ NumPy 不可用: {e}")

    print("-" * 30)

    # 测试pandas
    try:
        # 创建一个pandas DataFrame
        data = {'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'London', 'Paris']}
        df = pd.DataFrame(data)
        print("✓ Pandas 可用")
        print("  示例DataFrame:")
        print(df)
    except Exception as e:
        print(f"✗ Pandas 不可用: {e}")

    print("-" * 30)

    # 测试scipy
    try:
        # 创建一个简单的矩阵并保存为mat文件（在内存中）
        mat_data = {'array': np.array([[1, 2], [3, 4]])}
        print("✓ SciPy 可用")
        print(f"  示例矩阵: {mat_data['array']}")

        # 测试一些基本的scipy功能
        from scipy import linalg
        determinant = linalg.det(mat_data['array'])
        print(f"  矩阵行列式: {determinant:.2f}")
    except Exception as e:
        print(f"✗ SciPy 不可用: {e}")

    print("=" * 50)
    print("测试完成!")


# 读取.mat文件
def read_mat_file(file_path):
    """
    读取.mat文件并显示其内容
    """
    try:
        # 加载.mat文件
        mat_data = sio.loadmat(file_path)

        print("文件中的变量键:", list(mat_data.keys()))

        # 过滤掉系统变量（以__开头的）
        data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        print("有效数据变量:", data_keys)

        # 显示每个变量的信息
        for key in data_keys:
            data = mat_data[key]
            print(f"\n变量 '{key}':")
            print(f"  类型: {type(data)}")
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")

            if hasattr(data, 'shape') and len(data.shape) > 0:
                print(f"  总元素数: {data.size}")
                if data.size > 0:
                    print(f"  数据范围: [{np.min(data):.4f}, {np.max(data):.4f}]")

        return mat_data

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


# 读取并显示前100条数据
def display_first_100_data(mat_data):
    """
    显示.mat文件中的前100条数据
    """
    if mat_data is None:
        return

    # 找到包含数据的变量（通常是最大的数值数组）
    data_keys = [key for key in mat_data.keys() if not key.startswith('__')]

    for key in data_keys:
        data = mat_data[key]

        # 检查是否是数值数组
        if isinstance(data, np.ndarray) and data.size > 0:
            print(f"\n=== 变量 '{key}' 的前100条数据 ===")

            # 将数据展平为1维数组
            flattened_data = data.flatten()

            print(f"总数据条数: {len(flattened_data)}")

            # 显示前100条数据
            if len(flattened_data) >= 100:
                for i in range(100):
                    print(f"{i + 1}: {flattened_data[i]:.6f}")
            else:
                print("数据不足100条")
                for i in range(len(flattened_data)):
                    print(f"{i + 1}: {flattened_data[i]:.6f}")

            # 显示统计信息
            print(f"\n数据统计信息:")
            print(f"最小值: {np.min(flattened_data):.6f}")
            print(f"最大值: {np.max(flattened_data):.6f}")
            print(f"平均值: {np.mean(flattened_data):.6f}")
            print(f"标准差: {np.std(flattened_data):.6f}")


# 主程序
if __name__ == "__main__":
    #test_libraries()
    # 替换为您的.mat文件路径
    file_path = "data/target/A.mat"  # 或者您的具体文件名

    # 读取文件
    print(f"正在读取文件: {file_path}")
    mat_data = read_mat_file(file_path)

    # 显示前100条数据
    if mat_data is not None:
        display_first_100_data(mat_data)