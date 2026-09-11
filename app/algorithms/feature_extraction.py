"""
算法模块2：特征提取
技术方向：特征工程与模式识别
功能：时域特征、频域特征、时频域特征提取
"""
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt


class FeatureExtractor:
    """振动信号特征提取器"""

    def __init__(self, sample_rate=12000):
        self.sample_rate = sample_rate

    # ========== 时域特征 ==========
    def time_domain_features(self, data):
        """
        提取时域统计特征（16维）
        反映信号的幅值分布、冲击特性和能量特征
        """
        features = {}
        N = len(data)

        # 基本统计量
        features['mean'] = float(np.mean(data))
        features['std'] = float(np.std(data))
        features['rms'] = float(np.sqrt(np.mean(data ** 2)))  # 均方根
        features['peak'] = float(np.max(np.abs(data)))  # 峰值
        features['peak_to_peak'] = float(np.max(data) - np.min(data))  # 峰峰值
        features['skewness'] = float(skew(data))  # 偏度
        features['kurtosis'] = float(kurtosis(data))  # 峭度（故障敏感指标）
        features['variance'] = float(np.var(data))

        # 无量纲指标（对载荷变化不敏感，适合故障诊断）
        rms = features['rms']
        peak = features['peak']
        abs_mean = float(np.mean(np.abs(data)))

        features['crest_factor'] = float(peak / rms) if rms > 1e-10 else 0.0  # 峰值因子
        features['impulse_factor'] = float(peak / abs_mean) if abs_mean > 1e-10 else 0.0  # 脉冲因子
        features['margin_factor'] = float(peak / (np.mean(np.sqrt(np.abs(data))) ** 2)) if np.mean(np.sqrt(np.abs(data))) > 1e-10 else 0.0  # 裕度因子
        features['shape_factor'] = float(rms / abs_mean) if abs_mean > 1e-10 else 0.0  # 波形因子

        # 能量特征
        features['energy'] = float(np.sum(data ** 2))
        features['log_energy'] = float(np.log10(features['energy'] + 1e-10))

        return features

    # ========== 频域特征 ==========
    def frequency_domain_features(self, data):
        """
        提取频域特征（13维）
        反映信号的频率成分分布，不同故障有不同的特征频率
        """
        features = {}
        N = len(data)

        # FFT
        fft_vals = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.sample_rate)
        power_spectrum = np.abs(fft_vals) ** 2
        total_power = np.sum(power_spectrum)

        if total_power < 1e-10:
            total_power = 1e-10

        # 重心频率（频谱中心）
        features['spectral_centroid'] = float(np.sum(freqs * power_spectrum) / total_power)

        # 频谱方差
        centroid = features['spectral_centroid']
        features['spectral_variance'] = float(np.sum(((freqs - centroid) ** 2) * power_spectrum) / total_power)

        # 频谱峭度
        features['spectral_kurtosis'] = float(kurtosis(power_spectrum))

        # 频谱熵（衡量频率分布的无序程度）
        ps_norm = power_spectrum / total_power
        ps_norm = ps_norm[ps_norm > 0]
        features['spectral_entropy'] = float(-np.sum(ps_norm * np.log2(ps_norm)))

        # 主频（能量最大的频率）
        dominant_idx = np.argmax(power_spectrum)
        features['dominant_frequency'] = float(freqs[dominant_idx])
        features['dominant_power'] = float(power_spectrum[dominant_idx] / total_power)

        # 频带能量比（分5个频带）
        band_edges = [0, 500, 1000, 2000, 4000, self.sample_rate / 2]
        for i in range(len(band_edges) - 1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            band_power = np.sum(power_spectrum[mask])
            features[f'band_ratio_{i}'] = float(band_power / total_power)

        return features

    # ========== 时频域特征 ==========
    def time_frequency_features(self, data):
        """
        提取时频域特征（小波包能量熵，8维）
        技术点：小波包分解、能量熵
        """
        features = {}

        # 小波包3层分解，得到8个频带
        wp = pywt.WaveletPacket(data=data, wavelet='db4', mode='symmetric', maxlevel=3)
        level3_nodes = [node.path for node in wp.get_level(3, 'freq')]

        total_energy = 0
        band_energies = []

        for node_path in level3_nodes:
            coeffs = wp[node_path].data
            energy = np.sum(coeffs ** 2)
            band_energies.append(energy)
            total_energy += energy

        if total_energy < 1e-10:
            total_energy = 1e-10

        # 各频带能量比
        for i, energy in enumerate(band_energies):
            features[f'wp_energy_ratio_{i}'] = float(energy / total_energy)

        # 小波包能量熵
        energy_ratios = np.array(band_energies) / total_energy
        energy_ratios = energy_ratios[energy_ratios > 0]
        features['wp_energy_entropy'] = float(-np.sum(energy_ratios * np.log2(energy_ratios)))

        return features

    # ========== 包络谱特征（轴承故障专用） ==========
    def envelope_features(self, data):
        """
        包络分析特征（轴承故障诊断的经典方法）
        技术点：希尔伯特变换、包络谱
        """
        features = {}

        # 希尔伯特变换求解析信号
        analytic = signal.hilbert(data)
        envelope = np.abs(analytic)
        # 去除包络的直流分量
        envelope = envelope - np.mean(envelope)

        # 包络谱
        N = len(envelope)
        env_fft = np.abs(np.fft.rfft(envelope))
        env_freqs = np.fft.rfftfreq(N, d=1.0 / self.sample_rate)

        # 包络谱峰值（对应故障特征频率）
        peak_idx = np.argmax(env_fft)
        features['envelope_peak_freq'] = float(env_freqs[peak_idx])
        features['envelope_peak_amp'] = float(env_fft[peak_idx])

        # 包络谱峭度
        features['envelope_kurtosis'] = float(kurtosis(env_fft))

        return features

    def extract_all_features(self, data):
        """
        提取全部特征（时域16 + 频域13 + 时频8 + 包络4 = 41维）
        返回: 特征字典、特征名列表、特征值数组
        """
        all_features = {}
        all_features.update(self.time_domain_features(data))
        all_features.update(self.frequency_domain_features(data))
        all_features.update(self.time_frequency_features(data))
        all_features.update(self.envelope_features(data))

        feature_names = sorted(all_features.keys())
        feature_values = np.array([all_features[name] for name in feature_names])

        return all_features, feature_names, feature_values

    def extract_features_batch(self, frames):
        """
        批量提取多帧信号的特征
        frames: (n_frames, frame_size)
        返回: (n_frames, n_features) 特征矩阵
        """
        feature_matrix = []
        for frame in frames:
            _, _, values = self.extract_all_features(frame)
            feature_matrix.append(values)
        return np.array(feature_matrix)
