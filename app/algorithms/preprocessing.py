"""
算法模块1：数据预处理
技术方向：信号处理与数据清洗
功能：振动信号加载、降噪、归一化、分帧、异常值处理
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter
import pywt


class SignalPreprocessor:
    """振动信号预处理器"""

    def __init__(self, sample_rate=12000, frame_size=1024, hop_size=512):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

    def load_signal(self, file_path, column=None):
        """
        从CSV/TXT文件加载振动信号
        返回: numpy数组
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if column:
                data = df[column].values
            else:
                # 默认取第一列数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                data = df[numeric_cols[0]].values if len(numeric_cols) > 0 else df.iloc[:, 0].values
        elif file_path.endswith('.txt'):
            data = np.loadtxt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        return data.astype(np.float64)

    def remove_outliers(self, data, threshold=3.0):
        """
        基于3σ原则去除异常值，并用中位数填充
        """
        median = np.median(data)
        std = np.std(data)
        mask = np.abs(data - median) > threshold * std
        data_clean = data.copy()
        data_clean[mask] = median
        return data_clean, np.sum(mask)

    def denoise_wavelet(self, data, wavelet='db4', level=4, mode='soft'):
        """
        小波降噪：使用离散小波变换进行信号降噪
        技术点：多分辨率分析、阈值收缩
        """
        coeffs = pywt.wavedec(data, wavelet, level=level)
        # 计算通用阈值
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        # 对细节系数进行阈值处理
        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode=mode))
        # 重构信号
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        # 对齐长度
        denoised = denoised[:len(data)]
        return denoised

    def denoise_median(self, data, kernel_size=5):
        """中值滤波降噪，适用于脉冲噪声"""
        return median_filter(data, size=kernel_size)

    def denoise_bandpass(self, data, low_freq=10, high_freq=5000):
        """
        巴特沃斯带通滤波，去除低频漂移和高频噪声
        """
        nyq = 0.5 * self.sample_rate
        low = low_freq / nyq
        high = high_freq / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def normalize(self, data, method='zscore'):
        """
        信号归一化
        method: 'zscore' (零均值单位方差), 'minmax' (0-1), 'maxabs' (最大绝对值归一)
        """
        if method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std < 1e-10:
                return data - mean
            return (data - mean) / std
        elif method == 'minmax':
            dmin = np.min(data)
            dmax = np.max(data)
            if dmax - dmin < 1e-10:
                return np.zeros_like(data)
            return (data - dmin) / (dmax - dmin)
        elif method == 'maxabs':
            max_abs = np.max(np.abs(data))
            if max_abs < 1e-10:
                return data
            return data / max_abs
        else:
            raise ValueError(f"未知归一化方法: {method}")

    def framing(self, data):
        """
        信号分帧，返回帧数组 (n_frames, frame_size)
        """
        n_frames = 1 + (len(data) - self.frame_size) // self.hop_size
        frames = np.zeros((n_frames, self.frame_size))
        for i in range(n_frames):
            start = i * self.hop_size
            frames[i] = data[start:start + self.frame_size]
        return frames

    def preprocess_pipeline(self, data, denoise_method='wavelet', normalize_method='zscore'):
        """
        完整预处理流水线
        返回: 预处理后的信号、处理统计信息
        """
        stats = {}
        stats['original_length'] = len(data)
        stats['original_mean'] = float(np.mean(data))
        stats['original_std'] = float(np.std(data))

        # 1. 去除异常值
        data, outlier_count = self.remove_outliers(data)
        stats['outliers_removed'] = int(outlier_count)

        # 2. 降噪
        if denoise_method == 'wavelet':
            data = self.denoise_wavelet(data)
        elif denoise_method == 'median':
            data = self.denoise_median(data)
        elif denoise_method == 'bandpass':
            data = self.denoise_bandpass(data)
        stats['denoise_method'] = denoise_method

        # 3. 归一化
        data = self.normalize(data, method=normalize_method)
        stats['normalize_method'] = normalize_method
        stats['processed_length'] = len(data)

        return data, stats

    def get_spectrum(self, data):
        """获取信号频谱（用于可视化）"""
        n = len(data)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spectrum = np.abs(np.fft.rfft(data))
        return freqs, spectrum
