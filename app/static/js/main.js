// ========== 通用工具函数 ==========

// API请求封装
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    const mergedOptions = { ...defaultOptions, ...options };
    if (mergedOptions.body && typeof mergedOptions.body === 'object' && !(mergedOptions.body instanceof FormData)) {
        mergedOptions.body = JSON.stringify(mergedOptions.body);
    }
    if (mergedOptions.body instanceof FormData) {
        delete mergedOptions.headers['Content-Type'];
    }

    try {
        const response = await fetch(url, mergedOptions);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API请求失败:', error);
        showToast('请求失败: ' + error.message, 'error');
        return { success: false, error: error.message };
    }
}

// Toast提示
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.className = 'toast ' + type + ' show';
    setTimeout(() => {
        toast.className = 'toast ' + type;
    }, duration);
}

// 加载遮罩
function showLoading(text = '处理中...') {
    let overlay = document.getElementById('loadingOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = '<div class="loading"></div><div class="loading-text">' + text + '</div>';
        document.body.appendChild(overlay);
    } else {
        overlay.querySelector('.loading-text').textContent = text;
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// 格式化数字
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '-';
    return Number(num).toFixed(decimals);
}

// 格式化百分比
function formatPercent(num, decimals = 1) {
    if (num === null || num === undefined || isNaN(num)) return '-';
    return (Number(num) * 100).toFixed(decimals) + '%';
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 获取健康度标签
function getHealthTag(score) {
    if (score >= 80) return '<span class="tag tag-normal">优秀</span>';
    if (score >= 60) return '<span class="tag tag-info">良好</span>';
    if (score >= 40) return '<span class="tag tag-warning">注意</span>';
    if (score >= 20) return '<span class="tag tag-danger">恶化</span>';
    return '<span class="tag tag-danger">严重</span>';
}

// 获取故障类型标签
function getFaultTag(faultType) {
    if (faultType.includes('正常')) return '<span class="tag tag-normal">' + faultType + '</span>';
    return '<span class="tag tag-danger">' + faultType + '</span>';
}

// 初始化ECharts图表
function initChart(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return null;
    const chart = echarts.init(container);
    window.addEventListener('resize', () => chart.resize());
    return chart;
}

// 加载系统状态
async function loadSystemStatus() {
    try {
        const result = await apiRequest('/api/system/info');
        if (result.success && result.data) {
            const modelStatus = result.data.model_status;
            const currentModelEl = document.getElementById('currentModel');
            if (currentModelEl) {
                if (modelStatus.is_trained) {
                    currentModelEl.textContent = modelStatus.best_model || '已训练';
                } else {
                    currentModelEl.textContent = '未训练(规则模式)';
                }
            }
        }
    } catch (e) {
        console.error('加载系统状态失败:', e);
    }
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', () => {
    loadSystemStatus();
});
