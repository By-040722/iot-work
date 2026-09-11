"""
数据管理API路由
"""
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.services.data_service import DataService
from config import Config

data_bp = Blueprint('data', __name__)

_data_service = None


def get_data_service():
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@data_bp.route('/upload', methods=['POST'])
def upload_dataset():
    """上传数据集"""
    if 'file' not in request.files:
        return jsonify({'error': '未找到上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'不支持的文件格式'}), 400

    name = request.form.get('name', file.filename)
    description = request.form.get('description', '')

    service = get_data_service()
    try:
        dataset = service.save_dataset(file, name=name, description=description)
        return jsonify({'success': True, 'data': dataset})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@data_bp.route('/list', methods=['GET'])
def list_datasets():
    """获取数据集列表"""
    page = request.args.get('page', 1, type=int)
    size = request.args.get('size', 20, type=int)

    service = get_data_service()
    result = service.get_datasets(page=page, size=size)
    return jsonify({'success': True, 'data': result})


@data_bp.route('/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """获取数据集详情"""
    service = get_data_service()
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        return jsonify({'success': False, 'error': '数据集不存在'}), 404
    return jsonify({'success': True, 'data': dataset})


@data_bp.route('/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """删除数据集"""
    service = get_data_service()
    success = service.delete_dataset(dataset_id)
    if not success:
        return jsonify({'success': False, 'error': '数据集不存在'}), 404
    return jsonify({'success': True, 'message': '删除成功'})


@data_bp.route('/<int:dataset_id>/preview', methods=['GET'])
def preview_dataset(dataset_id):
    """预览数据集"""
    rows = request.args.get('rows', 10, type=int)
    service = get_data_service()
    result = service.preview_dataset(dataset_id, rows=rows)
    if result is None:
        return jsonify({'success': False, 'error': '数据集不存在'}), 404
    return jsonify({'success': True, 'data': result})
