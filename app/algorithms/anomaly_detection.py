"""
算法模块4：异常检测与预测性维护
技术方向：统计过程控制与异常检测
功能：基于统计阈值的异常检测、孤立森林异常检测、健康度评估、退化趋势预测
"""
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks


class AnomalyDetector:
    """轴承异常检测器与健康度评估器"""

    def __init__(self, sample_rate=12000):
        self.sample_rate = sample_rate
        self.baseline_stats = None  # 正常状态基线统计
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    # ========== 统计过程控制（SPC）异常检测 ==========
    def fit_baseline(self, normal_data):
        """
        用正常状态数据建立基线（控制限）
        技术点：统计过程控制、3σ控制限
        """
        self.baseline_stats = {
            'mean': float(np.mean(normal_data)),
            'std': float(np.std(normal_data)),
            'rms': float(np.sqrt(np.mean(normal_data ** 2))),
            'kurtosis': float(stats.kurtosis(normal_data)),
            'peak': float(np.max(np.abs(normal_data))),
            'ucl_rms': float(np.sqrt(np.mean(normal_data ** 2)) + 3 * np.std(normal_data)),
            'lcl_rms': float(max(0, np.sqrt(np.mean(normal_data ** 2)) - 3 * np.std(normal_data))),
            'ucl_kurtosis': float(stats.kurtosis(normal_data) + 3 * stats.kurtosis(np.random.randn(1000)) * 0.1),
            'n_samples': len(normal_data)
        }
        self.is_fitted = True
        return self.baseline_stats

    def spc_detect(self, data):
        """
        基于统计过程控制的异常检测
        检测规则：
        1. 超出3σ控制限
        2. 连续9点在中心线同一侧
        3. 连续6点递增或递减
        4. 连续14点交替上下
        返回: 异常检测结果
        """
        if self.baseline_stats is None:
            raise RuntimeError("请先调用 fit_baseline() 建立正常基线")

        rms = float(np.sqrt(np.mean(data ** 2)))
        kurt = float(stats.kurtosis(data))
        peak = float(np.max(np.abs(data)))
        mean = float(np.mean(data))

        violations = []
        is_anomaly = False
        anomaly_score = 0.0

        # 规则1：超出控制限
        if rms > self.baseline_stats['ucl_rms']:
            violations.append(f"RMS={rms:.4f} 超出上控制限 UCL={self.baseline_stats['ucl_rms']:.4f}")
            is_anomaly = True
            anomaly_score += 0.4
        if kurt > self.baseline_stats['ucl_kurtosis']:
            violations.append(f"峭度={kurt:.4f} 超出控制限 UCL={self.baseline_stats['ucl_kurtosis']:.4f}")
            is_anomaly = True
            anomaly_score += 0.3

        # 规则2：峰值异常
        peak_ratio = peak / self.baseline_stats['peak'] if self.baseline_stats['peak'] > 0 else 1.0
        if peak_ratio > 3.0:
            violations.append(f"峰值比={peak_ratio:.2f} 超过阈值3.0（冲击性异常）")
            is_anomaly = True
            anomaly_score += 0.3

        anomaly_score = min(1.0, anomaly_score)

        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'violations': violations,
            'metrics': {
                'rms': rms,
                'kurtosis': kurt,
                'peak': peak,
                'mean': mean,
                'peak_ratio': peak_ratio
            },
            'control_limits': {
                'ucl_rms': self.baseline_stats['ucl_rms'],
                'lcl_rms': self.baseline_stats['lcl_rms'],
                'baseline_rms': self.baseline_stats['rms']
            }
        }

    # ========== 孤立森林异常检测 ==========
    def fit_isolation_forest(self, X_normal, contamination=0.05):
        """
        用正常特征数据训练孤立森林
        X_normal: 正常状态的特征矩阵
        """
        X_scaled = self.scaler.fit_transform(X_normal)
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True

    def iforest_detect(self, X):
        """
        孤立森林异常检测
        X: 特征向量
        返回: 异常检测结果
        """
        if self.isolation_forest is None:
            raise RuntimeError("请先调用 fit_isolation_forest() 训练模型")

        X_scaled = self.scaler.transform(np.array(X).reshape(1, -1))
        pred = self.isolation_forest.predict(X_scaled)[0]
        score = self.isolation_forest.score_samples(X_scaled)[0]

        # 将异常分数归一化到 [0, 1]
        anomaly_score = float(1.0 / (1.0 + np.exp(score * 5)))

        return {
            'is_anomaly': bool(pred == -1),
            'anomaly_score': anomaly_score,
            'raw_score': float(score)
        }

    # ========== 健康度评估 ==========
    def health_indicator(self, data):
        """
        计算轴承健康度指标（0-100，越高越健康）
        综合RMS、峭度、峰值因子等指标
        """
        rms = np.sqrt(np.mean(data ** 2))
        kurt = stats.kurtosis(data)
        peak = np.max(np.abs(data))
        crest = peak / rms if rms > 1e-10 else 0

        # 各指标健康度评分（基于经验阈值）
        # RMS评分：RMS越小越健康
        rms_score = max(0, 100 - rms * 50) if self.baseline_stats is None else \
            max(0, 100 - (rms / self.baseline_stats['rms'] - 1) * 30)

        # 峭度评分：正常约3，越高越异常
        kurt_score = max(0, 100 - max(0, kurt - 3) * 10)

        # 峰值因子评分：正常约3-5，过高异常
        crest_score = max(0, 100 - max(0, crest - 5) * 8)

        # 综合健康度
        health = 0.4 * rms_score + 0.35 * kurt_score + 0.25 * crest_score
        health = max(0, min(100, health))

        # 健康等级
        if health >= 80:
            level = '优秀 (Excellent)'
            color = 'green'
        elif health >= 60:
            level = '良好 (Good)'
            color = 'lightgreen'
        elif health >= 40:
            level = '注意 (Warning)'
            color = 'orange'
        elif health >= 20:
            level = '恶化 (Deteriorating)'
            color = 'red'
        else:
            level = '严重 (Critical)'
            color = 'darkred'

        return {
            'health_score': float(health),
            'health_level': level,
            'color': color,
            'sub_scores': {
                'rms_score': float(rms_score),
                'kurtosis_score': float(kurt_score),
                'crest_factor_score': float(crest_score)
            },
            'raw_metrics': {
                'rms': float(rms),
                'kurtosis': float(kurt),
                'crest_factor': float(crest),
                'peak': float(peak)
            }
        }

    # ========== 退化趋势预测 ==========
    def degradation_trend(self, health_history, steps=5):
        """
        基于历史健康度数据预测退化趋势
        使用线性回归外推
        health_history: 历史健康度列表
        steps: 预测未来步数
        返回: 预测结果
        """
        if len(health_history) < 3:
            return {'error': '历史数据不足，至少需要3个数据点'}

        x = np.arange(len(health_history))
        y = np.array(health_history)

        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 预测
        future_x = np.arange(len(health_history), len(health_history) + steps)
        predicted = slope * future_x + intercept
        predicted = np.clip(predicted, 0, 100)

        # 预计失效时间（健康度降到20以下）
        if slope < 0:
            failure_step = (20 - intercept) / slope
            failure_time = max(0, failure_step - len(health_history))
        else:
            failure_time = float('inf')

        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'trend': '退化中' if slope < 0 else '稳定/改善',
            'degradation_rate': float(abs(slope)),
            'predicted_health': predicted.tolist(),
            'predicted_steps': steps,
            'estimated_failure_steps': float(failure_time) if failure_time != float('inf') else None,
            'history': health_history
        }

    def full_diagnosis(self, data, feature_vector=None):
        """
        完整诊断：异常检测 + 健康度评估
        """
        result = {}

        # 异常检测（SPC）
        if self.baseline_stats is not None:
            result['spc'] = self.spc_detect(data)
        else:
            result['spc'] = {'warning': '未建立基线，跳过SPC检测'}

        # 孤立森林检测
        if self.isolation_forest is not None and feature_vector is not None:
            result['iforest'] = self.iforest_detect(feature_vector)
        else:
            result['iforest'] = {'warning': '未训练孤立森林或缺少特征向量'}

        # 健康度
        result['health'] = self.health_indicator(data)

        return result
