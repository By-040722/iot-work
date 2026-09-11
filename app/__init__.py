"""
Flask应用工厂
"""
from flask import Flask
from flask_cors import CORS
from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    config_class.init_app(app)

    # 初始化数据库
    from app.models.database import db
    db.init_app(app)

    # 启用CORS
    CORS(app)

    # 注册蓝图
    from app.routes.main import main_bp
    from app.routes.diagnose import diagnose_bp
    from app.routes.data import data_bp
    from app.routes.model import model_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(diagnose_bp, url_prefix='/api/diagnose')
    app.register_blueprint(data_bp, url_prefix='/api/data')
    app.register_blueprint(model_bp, url_prefix='/api/model')

    # 创建数据库表
    with app.app_context():
        db.create_all()

    return app
