"""
算法模块3：故障分类
技术方向：机器学习与模式识别
功能：多模型故障分类（SVM、随机森林、KNN、决策树），模型训练、评估、预测
"""
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder


# 故障类型映射
FAULT_TYPES = {
    0: '正常 (Normal)',
    1: '内圈故障 (Inner Race Fault)',
    2: '外圈故障 (Outer Race Fault)',
    3: '滚动体故障 (Ball Fault)',
    4: '保持架故障 (Cage Fault)',
    5: '复合故障 (Compound Fault)'
}


class FaultClassifier:
    """轴承故障分类器（多模型集成）"""

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model_name = None
        self.is_trained = False

    def _get_model(self, model_type):
        """获取分类器实例"""
        models = {
            'svm': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=15,
                                                     min_samples_split=5, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'decision_tree': DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                             max_depth=5, random_state=42)
        }
        if model_type not in models:
            raise ValueError(f"不支持的模型类型: {model_type}，可选: {list(models.keys())}")
        return models[model_type]

    def train(self, X, y, model_types=None, test_size=0.3):
        """
        训练多个分类模型并选择最优模型
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组
        model_types: 要训练的模型列表，默认全部
        返回: 训练结果报告
        """
        if model_types is None:
            model_types = ['svm', 'random_forest', 'knn', 'decision_tree', 'gradient_boosting']

        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)

        # 划分训练集测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}
        best_score = 0

        for model_type in model_types:
            model = self._get_model(model_type)
            model.fit(X_train_scaled, y_train)

            # 预测
            y_pred = model.predict(X_test_scaled)

            # 评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

            results[model_type] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }

            self.models[model_type] = model

            if f1 > best_score:
                best_score = f1
                self.best_model_name = model_type

        self.is_trained = True

        # 保存模型
        self.save_models()

        return {
            'best_model': self.best_model_name,
            'best_f1': float(best_score),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'n_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'all_results': results
        }

    def predict(self, X, model_type=None):
        """
        对新样本进行故障预测
        X: 特征向量或特征矩阵
        返回: 预测标签、置信度、各类别概率
        """
        if not self.is_trained:
            # 尝试加载已保存模型
            self.load_models()
            if not self.is_trained:
                raise RuntimeError("模型未训练，请先训练或加载模型")

        if model_type is None:
            model_type = self.best_model_name

        model = self.models[model_type]
        X_scaled = self.scaler.transform(np.array(X).reshape(1, -1) if X.ndim == 1 else X)

        # 预测
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)

        # 解码标签
        predicted_labels = self.label_encoder.inverse_transform(y_pred)

        results = []
        for i in range(len(predicted_labels)):
            label = int(predicted_labels[i]) if isinstance(predicted_labels[i], (int, np.integer)) else predicted_labels[i]
            probs = y_proba[i].tolist()
            confidence = max(probs)
            class_probs = {
                FAULT_TYPES.get(int(c), str(c)): float(p)
                for c, p in zip(self.label_encoder.classes_, probs)
            }
            results.append({
                'predicted_label': label,
                'fault_type': FAULT_TYPES.get(label, str(label)),
                'confidence': float(confidence),
                'class_probabilities': class_probs
            })

        return results[0] if len(results) == 1 else results

    def feature_importance(self, feature_names=None):
        """获取特征重要性（基于随机森林）"""
        if 'random_forest' not in self.models:
            return None
        importances = self.models['random_forest'].feature_importances_
        if feature_names:
            return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return importances.tolist()

    def save_models(self):
        """保存所有模型到文件"""
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'best_model_name': self.best_model_name,
            'is_trained': self.is_trained
        }
        path = os.path.join(self.model_dir, 'fault_classifier.joblib')
        joblib.dump(save_data, path)
        return path

    def load_models(self):
        """从文件加载模型"""
        path = os.path.join(self.model_dir, 'fault_classifier.joblib')
        if not os.path.exists(path):
            return False
        save_data = joblib.load(path)
        self.models = save_data['models']
        self.scaler = save_data['scaler']
        self.label_encoder = save_data['label_encoder']
        self.best_model_name = save_data['best_model_name']
        self.is_trained = save_data['is_trained']
        return True

    def get_model_info(self):
        """获取模型信息"""
        return {
            'is_trained': self.is_trained,
            'best_model': self.best_model_name,
            'available_models': list(self.models.keys()) if self.models else [],
            'n_classes': len(self.label_encoder.classes_) if self.is_trained else 0,
            'class_names': self.label_encoder.classes_.tolist() if self.is_trained else [],
            'fault_types': FAULT_TYPES
        }
