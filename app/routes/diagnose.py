"""
诊断相关API路由
"""
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.services.diagnose_service import DiagnoseService
from config import Config

diagnose_bp = Blueprint('diagnose', __name__)

# 全局诊断服务实例
_diagnose_service = None


def get_diagnose_service():
    global _diagnose_service
    if _diagnose_service is None:
        _diagnose_service = DiagnoseService()
    return _diagnose_service


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@diagnose_bp.route('/upload', methods=['POST'])
def upload_and_diagnose():
    """
    上传振动信号文件并进行诊断
    """
    if 'file' not in request.files:
        return jsonify({'error': '未找到上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'不支持的文件格式，支持: {Config.ALLOWED_EXTENSIONS}'}), 400

    # 保存文件
    filename = secure_filename(file.filename)
    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        service = get_diagnose_service()
        result = service.diagnose_file(save_path, filename)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@diagnose_bp.route('/records', methods=['GET'])
def get_records():
    """获取诊断记录列表"""
    page = request.args.get('page', 1, type=int)
    size = request.args.get('size', 20, type=int)
    fault_type = request.args.get('fault_type', None)

    service = get_diagnose_service()
    result = service.get_records(page=page, size=size, fault_type=fault_type)
    return jsonify({'success': True, 'data': result})


@diagnose_bp.route('/record/<int:record_id>', methods=['GET'])
def get_record(record_id):
    """获取单条诊断记录详情"""
    service = get_diagnose_service()
    record = service.get_record(record_id)
    if not record:
        return jsonify({'success': False, 'error': '记录不存在'}), 404
    return jsonify({'success': True, 'data': record})


@diagnose_bp.route('/record/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    """删除诊断记录"""
    service = get_diagnose_service()
    success = service.delete_record(record_id)
    if not success:
        return jsonify({'success': False, 'error': '记录不存在'}), 404
    return jsonify({'success': True, 'message': '删除成功'})


@diagnose_bp.route('/stats', methods=['GET'])
def get_stats():
    """获取诊断统计信息"""
    service = get_diagnose_service()
    stats = service.get_statistics()
    return jsonify({'success': True, 'data': stats})
