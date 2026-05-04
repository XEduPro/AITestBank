from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
import shutil

# 设置根目录为当前文件所在目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 模拟用户数据库 - 用户名和手机号
users = {
    'u': '12345678900',
    '额德民': '12345678900',
    '贴斯特': '12345678900'
}

# 存储排行榜数据
leaderboard_file = 'leaderboard.csv'
if not os.path.exists(leaderboard_file):
    pd.DataFrame(columns=['username', 'phone', 'rmse', 'timestamp']).to_csv(leaderboard_file, index=False)


@app.route('/')
def index():
    leaderboard = pd.read_csv(leaderboard_file)
    leaderboard = leaderboard.sort_values('rmse', ascending=True)  # 按RMSE升序排序
    leaderboard = leaderboard.drop(columns=['phone'])
    return render_template('index.html', leaderboard=leaderboard.to_dict('records'))

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    phone = data.get('password')  # 这里password实际上是手机号
    
    if username in users and users[username] == phone:
        return jsonify({'success': True, 'phone': phone})
    return jsonify({'success': False, 'message': '用户名或手机号错误'})

@app.route('/upload', methods=['POST'])
def upload():
    if 'model_file' not in request.files or 'python_file' not in request.files:
        return jsonify({'success': False, 'message': '请上传模型文件和Python文件'})
    
    model_file = request.files['model_file']
    python_file = request.files['python_file']
    username = request.form.get('username')
    phone = request.form.get('phone')
    
    if model_file.filename == '' or python_file.filename == '':
        return jsonify({'success': False, 'message': '请选择所有必需的文件'})
    
    if not python_file.filename.endswith('.py'):
        return jsonify({'success': False, 'message': 'Python文件必须以.py结尾'})
    
    if not username or not phone:
        return jsonify({'success': False, 'message': '用户信息不完整'})
    
    # 创建用户专属文件夹
    user_folder = os.path.join(UPLOAD_FOLDER, f"{username}_{phone}")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
        # 创建data目录并复制测试集B.csv
        dst_data_dir = os.path.join(user_folder, 'data')
        os.makedirs(dst_data_dir, exist_ok=True)
        shutil.copy(os.path.join('data', '测试集B.csv'), os.path.join(dst_data_dir, '测试集B.csv'))
    
    # 保存上传的文件
    model_path = os.path.join(user_folder, model_file.filename)
    python_path = os.path.join(user_folder, f"{username}_{phone}.py")
    
    model_file.save(model_path)
    python_file.save(python_path)
    
    try:
        with open(python_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 只替换测试集路径
        content = content.replace('test_A_path = "data/测试集A.csv"', 'test_A_path = "data/测试集B.csv"')
        
        with open(python_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 切换到用户目录运行代码
        original_dir = os.getcwd()
        os.chdir(user_folder)
        
        # 运行推理代码
        subprocess.run(['python', os.path.basename(python_path)], capture_output=True, text=True)
        
        # 读取预测结果并计算RMSE
        rmse = 999.999  # 默认值
        if os.path.exists("预测结果.csv"):
            # 读取预测值
            y_pred_loaded = pd.read_csv("预测结果.csv")['预测值'].values
            # 读取真实值
            y_true = pd.read_csv('data/测试集B.csv')['BodyFat'].values
            # 计算RMSE
            rmse = np.sqrt(np.mean((y_true - y_pred_loaded) ** 2))
        
        # 切回原目录
        os.chdir(original_dir)

        # 先将新结果加入排行榜
        leaderboard = pd.read_csv(leaderboard_file)
        new_entry = pd.DataFrame({
            'username': [username],
            'phone': [phone],
            'rmse': [rmse],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)

        # 只保留每个用户最新一次提交
        leaderboard = leaderboard.sort_values('timestamp').drop_duplicates(['username', 'phone'], keep='last')
        leaderboard.to_csv(leaderboard_file, index=False)

        return jsonify({
            'success': True,
            'rmse': rmse,
            'message': '提交成功，后续会按照排名赋分'
        })

    except Exception as e:
        print('推理异常:', str(e))
        return jsonify({'success': False, 'message': f'处理文件时出错: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)