"""
主页面路由
"""
from flask import Blueprint, render_template, jsonify
from app.services.diagnose_service import DiagnoseService

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """仪表盘首页"""
    return render_template('index.html', active_page='dashboard')


@main_bp.route('/diagnose')
def diagnose_page():
    """在线诊断页面"""
    return render_template('diagnose.html', active_page='diagnose')


@main_bp.route('/data')
def data_page():
    """数据管理页面"""
    return render_template('data.html', active_page='data')


@main_bp.route('/model')
def model_page():
    """模型管理页面"""
    return render_template('model.html', active_page='model')


@main_bp.route('/history')
def history_page():
    """历史记录页面"""
    return render_template('history.html', active_page='history')


@main_bp.route('/api/system/info')
def system_info():
    """系统信息接口"""
    from app.services.model_service import ModelService
    model_service = ModelService()
    model_info = model_service.get_model_info()

    return jsonify({
        'system_name': '基于振动信号的智能故障诊断系统',
        'version': '1.0.0',
        'status': 'running',
        'model_status': model_info
    })


@main_bp.route('/api/system/health')
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'ok', 'message': 'System is running'})
