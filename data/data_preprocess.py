import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    """
    对原始轴承振动数据进行预处理：
    1. 数据清洗（去空值、去异常值）
    2. 标准化处理
    3. 划分训练集和测试集 (7:3)
    """
    # 1. 读取数据
    print(f"正在读取数据: {file_path}")
    df = pd.read_csv(file_path)

    # 2. 数据清洗
    # 删除缺失值
    df = df.dropna()
    # 简单去除异常值（超过3倍标准差的）
    df = df[(np.abs(df - df.mean()) <= (3 * df.std())).all(axis=1)]

    # 3. 特征和标签分离 (假设最后一列是标签，前面列是特征)
    # 注意：这里的列名需要根据你实际下载的数据集CSV表头进行修改！
    X = df.iloc[:, :-1].values 
    y = df.iloc[:, -1].values 

    # 4. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. 划分训练集和测试集 (7:3)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    print("数据预处理完成！")
    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

# 运行示例：把 your_data.csv 替换成你实际下载并放入 data 目录里的 CSV 文件名
# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = preprocess_data('your_data.csv')
