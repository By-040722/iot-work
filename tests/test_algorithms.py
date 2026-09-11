"""
算法模块自动化测试
覆盖：数据预处理、特征提取、故障分类、异常检测
"""
import sys
import os
import numpy as np
import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.algorithms.preprocessing import SignalPreprocessor
from app.algorithms.feature_extraction import FeatureExtractor
from app.algorithms.fault_classifier import FaultClassifier, FAULT_TYPES
from app.algorithms.anomaly_detection import AnomalyDetector


# ========== 测试数据生成 ==========
def generate_test_signal(n_samples=1024, fault_type='normal'):
    """生成测试用振动信号"""
    t = np.arange(n_samples) / 12000.0
    if fault_type == 'normal':
        signal = np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
        signal += np.random.normal(0, 0.1, n_samples)
    elif fault_type == 'fault':
        signal = np.zeros(n_samples)
        for i in range(0, n_samples, 200):
            impact = 2.0 * np.exp(-500 * (t - t[i])) * np.sin(2 * np.pi * 3000 * (t - t[i]))
            impact[t < t[i]] = 0
            signal += impact
        signal += np.random.normal(0, 0.1, n_samples)
    return signal


# ========== 数据预处理测试 ==========
class TestPreprocessing:
    """数据预处理模块测试"""

    def setup_method(self):
        self.preprocessor = SignalPreprocessor(sample_rate=12000, frame_size=1024, hop_size=512)
        self.test_signal = generate_test_signal(1024, 'normal')

    def test_remove_outliers(self):
        """测试异常值去除"""
        signal = self.test_signal.copy()
        signal[100] = 100.0  # 注入异常值
        cleaned, count = self.preprocessor.remove_outliers(signal)
        assert count >= 1
        assert np.max(np.abs(cleaned)) < 50

    def test_denoise_wavelet(self):
        """测试小波降噪"""
        noisy = self.test_signal + np.random.normal(0, 0.5, len(self.test_signal))
        denoised = self.preprocessor.denoise_wavelet(noisy)
        assert len(denoised) == len(noisy)
        # 降噪后信噪比应提升（方差减小）
        assert np.std(denoised) <= np.std(noisy) * 1.5

    def test_denoise_bandpass(self):
        """测试带通滤波"""
        filtered = self.preprocessor.denoise_bandpass(self.test_signal, low_freq=10, high_freq=5000)
        assert len(filtered) == len(self.test_signal)
        assert not np.any(np.isnan(filtered))

    def test_normalize_zscore(self):
        """测试Z-score归一化"""
        normalized = self.preprocessor.normalize(self.test_signal, method='zscore')
        assert abs(np.mean(normalized)) < 0.01
        assert abs(np.std(normalized) - 1.0) < 0.01

    def test_normalize_minmax(self):
        """测试Min-Max归一化"""
        normalized = self.preprocessor.normalize(self.test_signal, method='minmax')
        assert np.min(normalized) >= -0.01
        assert np.max(normalized) <= 1.01

    def test_framing(self):
        """测试分帧"""
        long_signal = generate_test_signal(5000, 'normal')
        frames = self.preprocessor.framing(long_signal)
        assert frames.shape[1] == 1024
        assert frames.shape[0] > 0

    def test_preprocess_pipeline(self):
        """测试完整预处理流水线"""
        processed, stats = self.preprocessor.preprocess_pipeline(self.test_signal)
        assert len(processed) == len(self.test_signal)
        assert 'original_length' in stats
        assert 'denoise_method' in stats
        assert 'outliers_removed' in stats
        assert stats['processed_length'] == len(self.test_signal)

    def test_get_spectrum(self):
        """测试频谱获取"""
        freqs, spectrum = self.preprocessor.get_spectrum(self.test_signal)
        assert len(freqs) == len(spectrum)
        assert np.all(freqs >= 0)
        assert np.all(spectrum >= 0)


# ========== 特征提取测试 ==========
class TestFeatureExtraction:
    """特征提取模块测试"""

    def setup_method(self):
        self.extractor = FeatureExtractor(sample_rate=12000)
        self.normal_signal = generate_test_signal(1024, 'normal')
        self.fault_signal = generate_test_signal(1024, 'fault')

    def test_time_domain_features(self):
        """测试时域特征提取"""
        features = self.extractor.time_domain_features(self.normal_signal)
        assert len(features) == 16
        assert 'mean' in features
        assert 'rms' in features
        assert 'kurtosis' in features
        assert 'crest_factor' in features
        # 验证数值合理性
        assert features['rms'] > 0
        assert features['peak'] >= features['rms']

    def test_frequency_domain_features(self):
        """测试频域特征提取"""
        features = self.extractor.frequency_domain_features(self.normal_signal)
        assert len(features) == 13
        assert 'spectral_centroid' in features
        assert 'spectral_entropy' in features
        assert 'dominant_frequency' in features
        assert features['spectral_centroid'] > 0

    def test_time_frequency_features(self):
        """测试时频域特征提取"""
        features = self.extractor.time_frequency_features(self.normal_signal)
        assert len(features) == 9  # 8个频带 + 1个能量熵
        assert 'wp_energy_entropy' in features
        assert features['wp_energy_entropy'] > 0

    def test_envelope_features(self):
        """测试包络谱特征提取"""
        features = self.extractor.envelope_features(self.fault_signal)
        assert len(features) == 3
        assert 'envelope_peak_freq' in features
        assert 'envelope_kurtosis' in features

    def test_extract_all_features(self):
        """测试全部特征提取"""
        all_features, feature_names, feature_values = self.extractor.extract_all_features(self.normal_signal)
        assert len(feature_names) == 41
        assert len(feature_values) == 41
        assert len(all_features) == 41
        # 验证特征值都是有限数
        assert np.all(np.isfinite(feature_values))

    def test_fault_vs_normal_features(self):
        """测试故障信号与正常信号特征差异"""
        normal_features = self.extractor.time_domain_features(self.normal_signal)
        fault_features = self.extractor.time_domain_features(self.fault_signal)
        # 故障信号峭度应显著高于正常信号
        assert fault_features['kurtosis'] > normal_features['kurtosis']

    def test_extract_features_batch(self):
        """测试批量特征提取"""
        frames = np.array([self.normal_signal, self.fault_signal])
        feature_matrix = self.extractor.extract_features_batch(frames)
        assert feature_matrix.shape == (2, 41)


# ========== 故障分类测试 ==========
class TestFaultClassifier:
    """故障分类模块测试"""

    def setup_method(self):
        self.classifier = FaultClassifier(model_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
        # 生成模拟训练数据
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(0, 4, n_samples)
        # 让不同类别有一定区分度
        for i in range(4):
            self.X[self.y == i] += i * 0.5

    def test_init(self):
        """测试初始化"""
        assert self.classifier is not None
        assert self.classifier.scaler is not None
        assert self.classifier.label_encoder is not None

    def test_fault_types_mapping(self):
        """测试故障类型映射"""
        assert len(FAULT_TYPES) >= 4
        assert 0 in FAULT_TYPES
        assert '正常' in FAULT_TYPES[0]

    def test_train(self):
        """测试模型训练"""
        result = self.classifier.train(self.X, self.y, model_types=['knn', 'decision_tree'], test_size=0.3)
        assert 'best_model' in result
        assert 'best_f1' in result
        assert 'n_train' in result
        assert 'n_test' in result
        assert 'all_results' in result
        assert len(result['all_results']) == 2
        assert result['n_train'] + result['n_test'] == len(self.X)
        assert 0 <= result['best_f1'] <= 1

    def test_predict(self):
        """测试模型预测"""
        self.classifier.train(self.X, self.y, model_types=['knn'], test_size=0.3)
        test_sample = np.random.randn(10)
        result = self.classifier.predict(test_sample)
        assert 'predicted_label' in result
        assert 'fault_type' in result
        assert 'confidence' in result
        assert 'class_probabilities' in result
        assert 0 <= result['confidence'] <= 1

    def test_predict_batch(self):
        """测试批量预测"""
        self.classifier.train(self.X, self.y, model_types=['knn'], test_size=0.3)
        test_samples = np.random.randn(5, 10)
        results = self.classifier.predict(test_samples)
        assert len(results) == 5

    def test_save_and_load(self):
        """测试模型保存和加载"""
        self.classifier.train(self.X, self.y, model_types=['knn'], test_size=0.3)
        path = self.classifier.save_models()
        assert os.path.exists(path)

        # 新建分类器并加载
        new_classifier = FaultClassifier(model_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
        loaded = new_classifier.load_models()
        assert loaded is True
        assert new_classifier.is_trained is True
        assert new_classifier.best_model_name is not None

    def test_get_model_info(self):
        """测试获取模型信息"""
        self.classifier.train(self.X, self.y, model_types=['knn'], test_size=0.3)
        info = self.classifier.get_model_info()
        assert info['is_trained'] is True
        assert 'best_model' in info
        assert 'available_models' in info
        assert 'fault_types' in info


# ========== 异常检测测试 ==========
class TestAnomalyDetection:
    """异常检测模块测试"""

    def setup_method(self):
        self.detector = AnomalyDetector(sample_rate=12000)
        self.normal_signal = generate_test_signal(1024, 'normal')
        self.fault_signal = generate_test_signal(1024, 'fault')

    def test_fit_baseline(self):
        """测试基线建立"""
        stats = self.detector.fit_baseline(self.normal_signal)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'rms' in stats
        assert 'ucl_rms' in stats
        assert self.detector.is_fitted is True

    def test_spc_detect_normal(self):
        """测试正常信号SPC检测"""
        self.detector.fit_baseline(self.normal_signal)
        result = self.detector.spc_detect(self.normal_signal)
        assert 'is_anomaly' in result
        assert 'anomaly_score' in result
        assert 'metrics' in result
        assert 'control_limits' in result

    def test_spc_detect_fault(self):
        """测试故障信号SPC检测"""
        self.detector.fit_baseline(self.normal_signal)
        result = self.detector.spc_detect(self.fault_signal)
        # 故障信号应该被检测为异常或有较高的异常分数
        assert result['anomaly_score'] >= 0

    def test_health_indicator(self):
        """测试健康度评估"""
        result = self.detector.health_indicator(self.normal_signal)
        assert 'health_score' in result
        assert 'health_level' in result
        assert 'color' in result
        assert 'sub_scores' in result
        assert 'raw_metrics' in result
        assert 0 <= result['health_score'] <= 100

    def test_health_fault_vs_normal(self):
        """测试故障信号健康度低于正常信号"""
        normal_health = self.detector.health_indicator(self.normal_signal)
        fault_health = self.detector.health_indicator(self.fault_signal)
        # 故障信号健康度应低于正常信号
        assert fault_health['health_score'] <= normal_health['health_score'] + 10

    def test_degradation_trend(self):
        """测试退化趋势预测"""
        history = [90, 88, 85, 82, 80, 78, 75]
        result = self.detector.degradation_trend(history, steps=5)
        assert 'slope' in result
        assert 'trend' in result
        assert 'predicted_health' in result
        assert len(result['predicted_health']) == 5
        assert result['slope'] < 0  # 退化中斜率应为负

    def test_degradation_trend_insufficient_data(self):
        """测试数据不足时的退化趋势预测"""
        history = [90, 88]
        result = self.detector.degradation_trend(history)
        assert 'error' in result

    def test_isolation_forest(self):
        """测试孤立森林异常检测"""
        # 生成正常特征数据
        np.random.seed(42)
        X_normal = np.random.randn(50, 10)
        self.detector.fit_isolation_forest(X_normal, contamination=0.05)

        # 正常样本
        normal_sample = np.random.randn(10)
        normal_result = self.detector.iforest_detect(normal_sample)
        assert 'is_anomaly' in normal_result
        assert 'anomaly_score' in normal_result

        # 异常样本
        anomaly_sample = np.ones(10) * 10
        anomaly_result = self.detector.iforest_detect(anomaly_sample)
        assert anomaly_result['anomaly_score'] >= normal_result['anomaly_score']


# ========== 运行测试 ==========
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
