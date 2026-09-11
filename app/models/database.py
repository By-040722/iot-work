"""
数据库模型定义
使用 Flask-SQLAlchemy ORM
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class DiagnosisRecord(db.Model):
    """诊断记录表"""
    __tablename__ = 'diagnosis_records'

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    fault_type = db.Column(db.String(100), nullable=False)
    fault_label = db.Column(db.Integer, default=0)
    confidence = db.Column(db.Float, default=0.0)
    health_score = db.Column(db.Float, default=100.0)
    health_level = db.Column(db.String(50), default='优秀')
    is_anomaly = db.Column(db.Boolean, default=False)
    anomaly_score = db.Column(db.Float, default=0.0)
    model_used = db.Column(db.String(50), default='random_forest')
    signal_length = db.Column(db.Integer, default=0)
    sample_rate = db.Column(db.Integer, default=12000)
    feature_summary = db.Column(db.Text)  # JSON字符串存储关键特征
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'file_name': self.file_name,
            'fault_type': self.fault_type,
            'fault_label': self.fault_label,
            'confidence': self.confidence,
            'health_score': self.health_score,
            'health_level': self.health_level,
            'is_anomaly': self.is_anomaly,
            'anomaly_score': self.anomaly_score,
            'model_used': self.model_used,
            'signal_length': self.signal_length,
            'sample_rate': self.sample_rate,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


class DatasetInfo(db.Model):
    """数据集信息表"""
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, default=0)
    n_samples = db.Column(db.Integer, default=0)
    n_features = db.Column(db.Integer, default=0)
    n_classes = db.Column(db.Integer, default=0)
    class_distribution = db.Column(db.Text)  # JSON
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'description': self.description,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


class ModelInfo(db.Model):
    """模型信息表"""
    __tablename__ = 'models'

    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, default=0.0)
    precision = db.Column(db.Float, default=0.0)
    recall = db.Column(db.Float, default=0.0)
    f1_score = db.Column(db.Float, default=0.0)
    cv_score = db.Column(db.Float, default=0.0)
    n_train = db.Column(db.Integer, default=0)
    n_test = db.Column(db.Integer, default=0)
    n_features = db.Column(db.Integer, default=0)
    n_classes = db.Column(db.Integer, default=0)
    is_best = db.Column(db.Boolean, default=False)
    trained_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'cv_score': self.cv_score,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'is_best': self.is_best,
            'trained_at': self.trained_at.strftime('%Y-%m-%d %H:%M:%S')
        }


class SystemLog(db.Model):
    """系统日志表"""
    __tablename__ = 'system_logs'

    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20), default='INFO')  # INFO, WARNING, ERROR
    module = db.Column(db.String(50))
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'level': self.level,
            'module': self.module,
            'message': self.message,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
