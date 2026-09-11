import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """系统配置"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'iot-bearing-diagnosis-secret-key')

    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(BASE_DIR, "bearing_diagnosis.db")}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 文件上传配置
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'mat', 'json'}

    # 模型存储路径
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    # 数据目录
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    # 算法参数
    SAMPLE_RATE = 12000  # 采样率 Hz
    FRAME_SIZE = 1024    # 帧长
    HOP_SIZE = 512       # 帧移

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
