"""
模型管理API路由
"""
from flask import Blueprint, request, jsonify
from app.services.model_service import ModelService

model_bp = Blueprint('model', __name__)

_model_service = None


def get_model_service():
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


@model_bp.route('/train', methods=['POST'])
def train_model():
    """训练模型"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求参数不能为空'}), 400

    dataset_id = data.get('dataset_id')
    if not dataset_id:
        return jsonify({'error': '请选择数据集'}), 400

    model_types = data.get('model_types', None)
    test_size = data.get('test_size', 0.3)

    service = get_model_service()
    try:
        result = service.train_model(dataset_id, model_types=model_types, test_size=test_size)
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 400
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@model_bp.route('/list', methods=['GET'])
def list_models():
    """获取模型列表"""
    service = get_model_service()
    result = service.get_models()
    return jsonify({'success': True, 'data': result})


@model_bp.route('/<int:model_id>', methods=['GET'])
def get_model(model_id):
    """获取模型详情"""
    service = get_model_service()
    model = service.get_model(model_id)
    if not model:
        return jsonify({'success': False, 'error': '模型不存在'}), 404
    return jsonify({'success': True, 'data': model})


@model_bp.route('/select', methods=['POST'])
def select_model():
    """设置默认模型"""
    data = request.get_json()
    model_type = data.get('model_type')
    if not model_type:
        return jsonify({'error': '请指定模型类型'}), 400

    service = get_model_service()
    result = service.select_model(model_type)
    if 'error' in result:
        return jsonify({'success': False, 'error': result['error']}), 400
    return jsonify({'success': True, 'data': result})


@model_bp.route('/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    """删除模型"""
    service = get_model_service()
    success = service.delete_model(model_id)
    if not success:
        return jsonify({'success': False, 'error': '模型不存在'}), 404
    return jsonify({'success': True, 'message': '删除成功'})


@model_bp.route('/feature-importance', methods=['GET'])
def feature_importance():
    """获取特征重要性"""
    service = get_model_service()
    result = service.get_feature_importance()
    if 'error' in result:
        return jsonify({'success': False, 'error': result['error']}), 400
    return jsonify({'success': True, 'data': result})


@model_bp.route('/info', methods=['GET'])
def model_info():
    """获取当前模型状态"""
    service = get_model_service()
    result = service.get_model_info()
    return jsonify({'success': True, 'data': result})
