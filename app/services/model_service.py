"""
模型服务层
封装模型训练、评估、管理相关的业务逻辑
"""
import os
import json
import numpy as np
from datetime import datetime
from app.models.database import db, ModelInfo, SystemLog
from app.algorithms.fault_classifier import FaultClassifier
from app.services.data_service import DataService
from config import Config


class ModelService:
    """模型管理服务"""

    def __init__(self):
        self.classifier = FaultClassifier(model_dir=Config.MODEL_DIR)
        self.data_service = DataService()
        # 尝试加载已训练模型
        self.classifier.load_models()

    def train_model(self, dataset_id, model_types=None, test_size=0.3):
        """
        训练模型
        """
        # 加载数据集
        X, y, feature_names = self.data_service.load_dataset_for_training(dataset_id)
        if X is None:
            return {'error': '数据集加载失败，请检查数据集格式'}

        if model_types is None:
            model_types = ['svm', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting']

        # 训练模型
        train_result = self.classifier.train(X, y, model_types=model_types, test_size=test_size)

        # 保存模型信息到数据库
        for model_type, metrics in train_result['all_results'].items():
            model_info = ModelInfo(
                model_type=model_type,
                model_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                cv_score=metrics['cv_mean'],
                n_train=train_result['n_train'],
                n_test=train_result['n_test'],
                n_features=train_result['n_features'],
                n_classes=train_result['n_classes'],
                is_best=(model_type == train_result['best_model'])
            )
            db.session.add(model_info)
        db.session.commit()

        # 记录日志
        log = SystemLog(
            level='INFO',
            module='model',
            message=f'模型训练完成: 最优模型={train_result["best_model"]}, F1={train_result["best_f1"]:.4f}'
        )
        db.session.add(log)
        db.session.commit()

        return train_result

    def get_models(self):
        """获取模型列表"""
        models = ModelInfo.query.order_by(ModelInfo.trained_at.desc()).all()
        return {
            'models': [m.to_dict() for m in models],
            'best_model': self.classifier.best_model_name,
            'is_trained': self.classifier.is_trained
        }

    def get_model(self, model_id):
        """获取模型详情"""
        model = ModelInfo.query.get(model_id)
        if not model:
            return None
        return model.to_dict()

    def select_model(self, model_type):
        """设置默认使用的模型"""
        if model_type not in self.classifier.models:
            return {'error': f'模型 {model_type} 不存在或未训练'}
        self.classifier.best_model_name = model_type
        # 更新数据库中的最优标记
        ModelInfo.query.update({'is_best': False})
        best = ModelInfo.query.filter_by(model_type=model_type).order_by(ModelInfo.trained_at.desc()).first()
        if best:
            best.is_best = True
        db.session.commit()
        # 保存模型
        self.classifier.save_models()
        return {'success': True, 'selected_model': model_type}

    def delete_model(self, model_id):
        """删除模型记录"""
        model = ModelInfo.query.get(model_id)
        if not model:
            return False
        db.session.delete(model)
        db.session.commit()
        return True

    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.classifier.is_trained:
            return {'error': '模型未训练'}
        importances = self.classifier.feature_importance()
        if importances is None:
            return {'error': '随机森林模型未训练'}
        return {
            'feature_importance': [
                {'feature': name, 'importance': float(imp)}
                for name, imp in importances
            ]
        }

    def get_model_info(self):
        """获取当前模型状态信息"""
        return self.classifier.get_model_info()

    def retrain_with_features(self, X, y, feature_names=None):
        """
        使用特征矩阵直接训练模型（用于模拟数据训练）
        """
        train_result = self.classifier.train(
            X, y,
            model_types=['svm', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting'],
            test_size=0.3
        )

        # 保存到数据库
        for model_type, metrics in train_result['all_results'].items():
            model_info = ModelInfo(
                model_type=model_type,
                model_name=f"{model_type}_simulated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                cv_score=metrics['cv_mean'],
                n_train=train_result['n_train'],
                n_test=train_result['n_test'],
                n_features=train_result['n_features'],
                n_classes=train_result['n_classes'],
                is_best=(model_type == train_result['best_model'])
            )
            db.session.add(model_info)
        db.session.commit()

        return train_result
