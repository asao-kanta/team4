from flask import Flask, render_template, request, redirect, url_for, send_from_directory

import numpy as np

from PIL import Image

import tensorflow as tf
import keras
from keras.models import model_from_json

from keras.preprocessing.image import img_to_array, load_img

import os
import glob
from secrets import token_hex
from werkzeug.utils import secure_filename

app = Flask(__name__)

def img_pred(image):
    # モデル読み込み
    with open('./model/and.json', 'r') as f:
        json_string = f.read()
    model = model_from_json(json_string)

    model.load_weights('./model/and_weight.hdf5')

    image = image.convert("RGB")
    #image = image.resize((120, 120))

    image_diverse = np.asarray(image)

    # 予測
    preds = model.predict(image_diverse.reshape(1, 120, 120, 3)).argmax()

    label_name = {
        'Clear plastic bottle':0,
        'Disposable plastic cup':1,
        'Drink can':2, 
        'Glass bottle':3, 
        'Plastic film':4
    }

    result = [k for k, v in label_name.items() if v == preds]

    return result[0]

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/result', methods=['POST'])
def result():
    # submitした画像が存在したら処理する
    if request.files['image']:
        # 画像の読み込み
        image_file = request.files['image']
        image_name = secure_filename(image_file.filename)
        image_file.save(image_name)

        # ファイル名をtokenに
        hex = token_hex(16)
        _, ext = os.path.splitext(image_name)
        new_file_name = f'{hex}{ext}'
        os.rename(image_name, new_file_name)

        image_load = load_img(new_file_name, target_size=(120,120))

        # クラスの予測をする関数の実行
        predict_Confidence = img_pred(image_load)
        
        # 現在のディレクトリのやつは削除
        os.remove(new_file_name)

        # render_template('./result.html')
        return render_template('./result.html', predict_Confidence=predict_Confidence)


if __name__ == '__main__':
    app.run()
