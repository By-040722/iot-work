"""
轴承振动信号模拟数据集生成脚本
基于轴承故障振动信号数学模型生成模拟数据
生成4种状态：正常、内圈故障、外圈故障、滚动体故障
"""
import numpy as np
import pandas as pd
import os


def generate_bearing_signal(fault_type='normal', n_samples=1024, sample_rate=12000,
                             rotational_speed=1800, noise_level=0.1, seed=None):
    """
    生成单条轴承振动信号

    参数:
        fault_type: 故障类型 ('normal', 'inner', 'outer', 'ball')
        n_samples: 采样点数
        sample_rate: 采样率 (Hz)
        rotational_speed: 转速 (RPM)
        noise_level: 噪声水平
        seed: 随机种子

    返回:
        振动信号数组 (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples) / sample_rate
    fr = rotational_speed / 60.0  # 转频 (Hz)

    # 轴承参数（典型深沟球轴承）
    n_balls = 9          # 滚动体数量
    ball_diameter = 7.94  # 滚动体直径 (mm)
    pitch_diameter = 34.5  # 节圆直径 (mm)
    contact_angle = 0      # 接触角

    # 故障特征频率
    bpfi = 0.5 * fr * n_balls * (1 + ball_diameter / pitch_diameter * np.cos(contact_angle))  # 内圈
    bpfo = 0.5 * fr * n_balls * (1 - ball_diameter / pitch_diameter * np.cos(contact_angle))  # 外圈
    bsf = 0.5 * fr * pitch_diameter / ball_diameter * (1 - (ball_diameter / pitch_diameter * np.cos(contact_angle)) ** 2)  # 滚动体

    signal = np.zeros(n_samples)

    # 正常信号：旋转频率及其谐波 + 随机噪声
    if fault_type == 'normal':
        for harmonic in range(1, 4):
            amplitude = 1.0 / harmonic
            signal += amplitude * np.sin(2 * np.pi * harmonic * fr * t + np.random.uniform(0, 2*np.pi))
        # 固有频率成分
        signal += 0.3 * np.sin(2 * np.pi * 2500 * t)
        signal += 0.2 * np.sin(2 * np.pi * 3500 * t)

    # 内圈故障：以BPFI为间隔的周期性冲击，被转频调制
    elif fault_type == 'inner':
        impact_interval = 1.0 / bpfi
        n_impacts = int(n_samples / sample_rate / impact_interval) + 2
        for i in range(n_impacts):
            impact_time = i * impact_interval
            # 转频调制（内圈故障点受载荷区影响）
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * fr * impact_time)
            amplitude = 2.0 * modulation
            # 衰减振荡冲击
            impact = amplitude * np.exp(-500 * (t - impact_time)) * \
                     np.sin(2 * np.pi * 3000 * (t - impact_time)) * \
                     (t >= impact_time)
            signal += impact
        # 基础旋转成分
        signal += 0.5 * np.sin(2 * np.pi * fr * t)

    # 外圈故障：以BPFO为间隔的周期性冲击（幅值稳定）
    elif fault_type == 'outer':
        impact_interval = 1.0 / bpfo
        n_impacts = int(n_samples / sample_rate / impact_interval) + 2
        for i in range(n_impacts):
            impact_time = i * impact_interval
            amplitude = 2.5
            impact = amplitude * np.exp(-600 * (t - impact_time)) * \
                     np.sin(2 * np.pi * 2800 * (t - impact_time)) * \
                     (t >= impact_time)
            signal += impact
        signal += 0.4 * np.sin(2 * np.pi * fr * t)

    # 滚动体故障：以BSF为间隔的冲击，被保持架频率调制
    elif fault_type == 'ball':
        impact_interval = 1.0 / bsf
        n_impacts = int(n_samples / sample_rate / impact_interval) + 2
        cage_freq = 0.5 * fr * (1 - ball_diameter / pitch_diameter * np.cos(contact_angle))
        for i in range(n_impacts):
            impact_time = i * impact_interval
            # 保持架调制
            modulation = 0.6 + 0.4 * np.sin(2 * np.pi * cage_freq * impact_time)
            amplitude = 1.8 * modulation
            impact = amplitude * np.exp(-400 * (t - impact_time)) * \
                     np.sin(2 * np.pi * 3200 * (t - impact_time)) * \
                     (t >= impact_time)
            signal += impact
        signal += 0.5 * np.sin(2 * np.pi * fr * t)

    # 添加高斯白噪声
    noise = np.random.normal(0, noise_level * np.std(signal) if np.std(signal) > 0 else noise_level, n_samples)
    signal += noise

    # 归一化
    signal = signal / (np.max(np.abs(signal)) + 1e-10)

    return signal


def generate_dataset(output_path, n_per_class=500, n_samples=1024, sample_rate=12000):
    """
    生成完整数据集并保存为CSV

    参数:
        output_path: 输出文件路径
        n_per_class: 每类样本数
        n_samples: 每条样本的采样点数
        sample_rate: 采样率
    """
    fault_types = {
        'normal': 0,
        'inner': 1,
        'outer': 2,
        'ball': 3
    }
    fault_names = {
        0: '正常',
        1: '内圈故障',
        2: '外圈故障',
        3: '滚动体故障'
    }

    all_data = []
    all_labels = []
    all_names = []

    print(f"开始生成数据集，每类 {n_per_class} 条样本，每条 {n_samples} 点...")

    for fault_type, label in fault_types.items():
        print(f"  生成 {fault_names[label]} ({fault_type}) 数据...")
        for i in range(n_per_class):
            signal = generate_bearing_signal(
                fault_type=fault_type,
                n_samples=n_samples,
                sample_rate=sample_rate,
                rotational_speed=np.random.uniform(1500, 2100),  # 转速随机变化
                noise_level=np.random.uniform(0.05, 0.15),       # 噪声水平随机
                seed=i + label * 10000
            )
            all_data.append(signal)
            all_labels.append(label)
            all_names.append(fault_names[label])

    # 构建DataFrame
    columns = [f'sample_{i}' for i in range(n_samples)]
    df = pd.DataFrame(all_data, columns=columns)
    df.insert(0, 'fault_name', all_names)
    df.insert(0, 'fault_type', all_labels)

    # 打乱顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n数据集生成完成！")
    print(f"  文件路径: {output_path}")
    print(f"  总样本数: {len(df)}")
    print(f"  特征数: {n_samples}")
    print(f"  类别分布:")
    for label, name in fault_names.items():
        count = len(df[df['fault_type'] == label])
        print(f"    {name}: {count} 条")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return df


if __name__ == '__main__':
    # 生成主数据集（用于模型训练）
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    generate_dataset(
        output_path=os.path.join(output_dir, 'bearing_dataset.csv'),
        n_per_class=300,
        n_samples=1024,
        sample_rate=12000
    )

    # 生成单条测试信号（用于在线诊断演示）
    print("\n生成单条测试信号...")
    test_signal = generate_bearing_signal('outer', n_samples=10000, sample_rate=12000, seed=42)
    test_df = pd.DataFrame({'vibration': test_signal})
    test_df.to_csv(os.path.join(output_dir, 'test_signal_outer.csv'), index=False)
    print(f"  测试信号已保存: test_signal_outer.csv (10000点)")

    test_signal2 = generate_bearing_signal('normal', n_samples=10000, sample_rate=12000, seed=43)
    test_df2 = pd.DataFrame({'vibration': test_signal2})
    test_df2.to_csv(os.path.join(output_dir, 'test_signal_normal.csv'), index=False)
    print(f"  测试信号已保存: test_signal_normal.csv (10000点)")
