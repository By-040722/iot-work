"""
数据服务层
封装数据集管理相关的业务逻辑
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from app.models.database import db, DatasetInfo, SystemLog
from config import Config


class DataService:
    """数据管理服务"""

    def __init__(self):
        self.upload_dir = Config.UPLOAD_FOLDER
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_dataset(self, file_storage, name=None, description=''):
        """
        保存上传的数据集文件
        """
        original_name = file_storage.filename
        if not name:
            name = original_name

        # 生成保存路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = f"{timestamp}_{original_name}"
        save_path = os.path.join(self.upload_dir, safe_name)
        file_storage.save(save_path)

        # 解析数据集信息
        file_size = os.path.getsize(save_path)
        n_samples = 0
        n_features = 0
        n_classes = 0
        class_distribution = {}

        try:
            if original_name.endswith('.csv'):
                df = pd.read_csv(save_path)
                n_samples = len(df)
                n_features = len(df.columns) - 1  # 假设最后一列是标签
                # 尝试获取类别分布
                label_col = df.columns[-1]
                class_distribution = df[label_col].value_counts().to_dict()
                n_classes = len(class_distribution)
        except Exception as e:
            # 解析失败不影响保存
            pass

        # 保存到数据库
        dataset = DatasetInfo(
            name=name,
            file_path=save_path,
            file_size=file_size,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_distribution=json.dumps(class_distribution, ensure_ascii=False),
            description=description
        )
        db.session.add(dataset)
        db.session.commit()

        # 记录日志
        log = SystemLog(
            level='INFO',
            module='data',
            message=f'数据集上传: {name}, 样本数: {n_samples}, 特征数: {n_features}'
        )
        db.session.add(log)
        db.session.commit()

        return dataset.to_dict()

    def get_datasets(self, page=1, size=20):
        """获取数据集列表"""
        pagination = DatasetInfo.query.order_by(DatasetInfo.created_at.desc()).paginate(
            page=page, per_page=size, error_out=False
        )
        return {
            'datasets': [d.to_dict() for d in pagination.items],
            'total': pagination.total,
            'page': page,
            'size': size,
            'pages': pagination.pages
        }

    def get_dataset(self, dataset_id):
        """获取数据集详情"""
        dataset = DatasetInfo.query.get(dataset_id)
        if not dataset:
            return None
        result = dataset.to_dict()
        if dataset.class_distribution:
            result['class_distribution'] = json.loads(dataset.class_distribution)
        return result

    def delete_dataset(self, dataset_id):
        """删除数据集"""
        dataset = DatasetInfo.query.get(dataset_id)
        if not dataset:
            return False
        # 删除文件
        if os.path.exists(dataset.file_path):
            try:
                os.remove(dataset.file_path)
            except Exception:
                pass
        db.session.delete(dataset)
        db.session.commit()
        return True

    def preview_dataset(self, dataset_id, rows=10):
        """预览数据集前N行"""
        dataset = DatasetInfo.query.get(dataset_id)
        if not dataset:
            return None
        if not os.path.exists(dataset.file_path):
            return {'error': '文件不存在'}
        try:
            df = pd.read_csv(dataset.file_path, nrows=rows)
            return {
                'columns': df.columns.tolist(),
                'data': df.values.tolist(),
                'preview_rows': min(rows, len(df)),
                'total_rows': dataset.n_samples
            }
        except Exception as e:
            return {'error': f'预览失败: {str(e)}'}

    def load_dataset_for_training(self, dataset_id):
        """
        加载数据集用于模型训练
        返回: X (特征矩阵), y (标签数组), feature_names
        """
        dataset = DatasetInfo.query.get(dataset_id)
        if not dataset:
            return None, None, None
        if not os.path.exists(dataset.file_path):
            return None, None, None

        df = pd.read_csv(dataset.file_path)
        # 假设最后一列是标签，前面是特征
        # 排除非数值列
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None, None, None

        X = numeric_df.iloc[:, :-1].values
        y = numeric_df.iloc[:, -1].values
        feature_names = numeric_df.columns[:-1].tolist()

        return X, y, feature_names
