"""
诊断服务层
封装诊断相关的业务逻辑，协调各算法模块
"""
import os
import json
import numpy as np
from datetime import datetime
from app.models.database import db, DiagnosisRecord, SystemLog
from app.algorithms.preprocessing import SignalPreprocessor
from app.algorithms.feature_extraction import FeatureExtractor
from app.algorithms.fault_classifier import FaultClassifier
from app.algorithms.anomaly_detection import AnomalyDetector
from config import Config


class DiagnoseService:
    """诊断服务"""

    def __init__(self):
        self.preprocessor = SignalPreprocessor(
            sample_rate=Config.SAMPLE_RATE,
            frame_size=Config.FRAME_SIZE,
            hop_size=Config.HOP_SIZE
        )
        self.feature_extractor = FeatureExtractor(sample_rate=Config.SAMPLE_RATE)
        self.classifier = FaultClassifier(model_dir=Config.MODEL_DIR)
        self.anomaly_detector = AnomalyDetector(sample_rate=Config.SAMPLE_RATE)
        # 尝试加载已训练模型
        self.classifier.load_models()

    def diagnose_file(self, file_path, file_name):
        """
        对上传的振动信号文件进行完整诊断
        返回: 诊断结果字典
        """
        try:
            # 1. 加载信号
            raw_signal = self.preprocessor.load_signal(file_path)
            signal_length = len(raw_signal)

            # 2. 预处理
            processed_signal, preprocess_stats = self.preprocessor.preprocess_pipeline(
                raw_signal, denoise_method='wavelet', normalize_method='zscore'
            )

            # 3. 分帧
            frames = self.preprocessor.framing(processed_signal)

            # 4. 特征提取（取所有帧的平均特征）
            all_features = []
            for frame in frames:
                _, _, feature_values = self.feature_extractor.extract_all_features(frame)
                all_features.append(feature_values)
            feature_matrix = np.array(all_features)
            avg_features = np.mean(feature_matrix, axis=0)

            # 获取特征名
            _, feature_names, _ = self.feature_extractor.extract_all_features(frames[0])
            feature_dict = {name: float(val) for name, val in zip(feature_names, avg_features)}

            # 5. 故障分类
            if self.classifier.is_trained:
                classify_result = self.classifier.predict(avg_features)
                fault_type = classify_result['fault_type']
                fault_label = classify_result['predicted_label']
                confidence = classify_result['confidence']
                class_probabilities = classify_result['class_probabilities']
                model_used = self.classifier.best_model_name
            else:
                # 未训练模型时，基于规则的简单诊断
                fault_type, fault_label, confidence = self._rule_based_diagnose(processed_signal)
                class_probabilities = {fault_type: confidence}
                model_used = 'rule_based'

            # 6. 异常检测与健康度
            health_result = self.anomaly_detector.health_indicator(processed_signal)
            is_anomaly = health_result['health_score'] < 60
            anomaly_score = float(1.0 - health_result['health_score'] / 100.0)

            # 7. 获取频谱数据（用于可视化）
            freqs, spectrum = self.preprocessor.get_spectrum(processed_signal)
            # 降采样，减少数据量
            max_points = 500
            if len(freqs) > max_points:
                indices = np.linspace(0, len(freqs) - 1, max_points, dtype=int)
                freqs = freqs[indices]
                spectrum = spectrum[indices]

            # 8. 保存诊断记录
            record = DiagnosisRecord(
                file_name=file_name,
                fault_type=fault_type,
                fault_label=fault_label,
                confidence=confidence,
                health_score=health_result['health_score'],
                health_level=health_result['health_level'],
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                model_used=model_used,
                signal_length=signal_length,
                sample_rate=Config.SAMPLE_RATE,
                feature_summary=json.dumps({k: round(v, 6) for k, v in feature_dict.items()})
            )
            db.session.add(record)
            db.session.commit()

            # 9. 记录日志
            log = SystemLog(
                level='INFO',
                module='diagnose',
                message=f'诊断完成: {file_name}, 结果: {fault_type}, 置信度: {confidence:.4f}'
            )
            db.session.add(log)
            db.session.commit()

            # 10. 组装返回结果
            # 波形数据降采样
            waveform_points = 1000
            if len(processed_signal) > waveform_points:
                w_indices = np.linspace(0, len(processed_signal) - 1, waveform_points, dtype=int)
                waveform = processed_signal[w_indices].tolist()
                raw_waveform = raw_signal[w_indices].tolist()
            else:
                waveform = processed_signal.tolist()
                raw_waveform = raw_signal.tolist()

            result = {
                'record_id': record.id,
                'file_name': file_name,
                'signal_length': signal_length,
                'sample_rate': Config.SAMPLE_RATE,
                'preprocess_stats': preprocess_stats,
                'diagnosis': {
                    'fault_type': fault_type,
                    'fault_label': fault_label,
                    'confidence': confidence,
                    'class_probabilities': class_probabilities,
                    'model_used': model_used
                },
                'health': health_result,
                'anomaly': {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score
                },
                'features': {
                    'names': feature_names,
                    'values': avg_features.tolist(),
                    'dict': {k: round(v, 6) for k, v in feature_dict.items()}
                },
                'visualization': {
                    'raw_waveform': raw_waveform,
                    'processed_waveform': waveform,
                    'frequencies': freqs.tolist(),
                    'spectrum': spectrum.tolist()
                },
                'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }

            return result

        except Exception as e:
            # 记录错误日志
            log = SystemLog(
                level='ERROR',
                module='diagnose',
                message=f'诊断失败: {file_name}, 错误: {str(e)}'
            )
            db.session.add(log)
            db.session.commit()
            raise

    def _rule_based_diagnose(self, signal):
        """
        基于规则的简单诊断（当模型未训练时使用）
        基于峭度和峰值因子判断故障
        """
        from scipy.stats import kurtosis
        kurt = kurtosis(signal)
        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))
        crest = peak / rms if rms > 1e-10 else 0

        if kurt < 4 and crest < 5:
            return '正常 (Normal)', 0, 0.85
        elif 4 <= kurt < 8:
            return '外圈故障 (Outer Race Fault)', 2, 0.70
        elif 8 <= kurt < 15:
            return '内圈故障 (Inner Race Fault)', 1, 0.65
        else:
            return '滚动体故障 (Ball Fault)', 3, 0.60

    def get_records(self, page=1, size=20, fault_type=None):
        """获取诊断记录列表"""
        query = DiagnosisRecord.query
        if fault_type:
            query = query.filter(DiagnosisRecord.fault_type.like(f'%{fault_type}%'))
        query = query.order_by(DiagnosisRecord.created_at.desc())
        pagination = query.paginate(page=page, per_page=size, error_out=False)
        return {
            'records': [r.to_dict() for r in pagination.items],
            'total': pagination.total,
            'page': page,
            'size': size,
            'pages': pagination.pages
        }

    def get_record(self, record_id):
        """获取单条诊断记录详情"""
        record = DiagnosisRecord.query.get(record_id)
        if not record:
            return None
        result = record.to_dict()
        if record.feature_summary:
            result['features'] = json.loads(record.feature_summary)
        return result

    def delete_record(self, record_id):
        """删除诊断记录"""
        record = DiagnosisRecord.query.get(record_id)
        if not record:
            return False
        db.session.delete(record)
        db.session.commit()
        return True

    def get_statistics(self):
        """获取诊断统计信息"""
        total = DiagnosisRecord.query.count()
        anomaly_count = DiagnosisRecord.query.filter_by(is_anomaly=True).count()
        normal_count = DiagnosisRecord.query.filter(DiagnosisRecord.fault_type.like('%正常%')).count()

        # 按故障类型统计
        fault_stats = {}
        records = DiagnosisRecord.query.all()
        for r in records:
            ftype = r.fault_type.split(' ')[0] if ' ' in r.fault_type else r.fault_type
            fault_stats[ftype] = fault_stats.get(ftype, 0) + 1

        # 平均健康度
        avg_health = float(np.mean([r.health_score for r in records])) if records else 100.0

        return {
            'total_diagnosis': total,
            'anomaly_count': anomaly_count,
            'anomaly_rate': float(anomaly_count / total) if total > 0 else 0,
            'normal_count': normal_count,
            'avg_health_score': round(avg_health, 2),
            'fault_distribution': fault_stats
        }
