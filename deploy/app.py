from flask import Flask  # 导入Flask包
from flask_script import Manager
from flask import render_template
from werkzeug.utils import secure_filename
from flask import request
import subprocess
import os

app = Flask(__name__)  # 获取Flask对象，以当前模块名为参数
manager = Manager(app)

# 路由默认为（127.0.0.1:5000）
@app.route('/')  # 装饰器对该方法进行路由设置，请求的地址
def hello_world():  # 方法名称
    name = 'TengTengCai'
    # print(1/0)
    # return 'Hello World!'  # 返回响应的内容
    return render_template('post.html')

@app.route("/fileupload", methods=['POST'])
def post():
    f = request.files.get('fileupload')
    print(f.filename)
    f.save('./test_img/' + secure_filename(f.filename))
    path = '/home/ouc/Flask/test_img/' + secure_filename(f.filename)
    cmd = 'python test.py --path ' + path
    print(cmd)
    a = subprocess.Popen(cmd,
                            stdin = subprocess.PIPE,
                            stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE,shell=True,universal_newlines=True)
    out, err = a.communicate()
    # return out
    return render_template('prediction_result.html',predicted_class = '1')

if __name__ == '__main__':
    manager.run()