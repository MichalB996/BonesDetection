import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import pathlib
from inference_new import compute

pa = pathlib.Path(__file__).parent.absolute()
if not os.path.exists(str(pa)+'\images'):
    os.makedirs(str(pa)+'\images')
imagePath = str(pa)+'\images'


app = Flask(__name__)

@app.route('/results/<filename>')
def send_file(filename):
    return send_from_directory(str(pa)+'\images', filename)

@app.route('/', methods=['GET', 'POST'])
def mainPage():
    if request.method == 'GET':
        return render_template('home_page.html')
    if request.method == 'POST':
        if request.form['startModel'] == 'StartComputation':
            file_image = request.files['image']
            file_mask = request.files['mask']
            if file_image.filename != '':
                filename_image = secure_filename(file_image.filename)
                image_path = os.path.join(imagePath, filename_image)
                file_image.save(image_path)
                file = open(image_path)
                # convert bfile to uint8
                nparr_image = np.fromfile(file, np.uint8)
                # decode image
                image = cv2.imdecode(nparr_image, cv2.IMREAD_REDUCED_GRAYSCALE_8)
                image = cv2.resize(image, (256, 512))

                filename_mask = secure_filename(file_mask.filename)
                mask_path = os.path.join(imagePath, filename_mask)
                file_mask.save(mask_path)
                file = open(mask_path)
                # convert file to uint8
                nparr_mask = np.fromfile(file, np.uint8)
                # decode mask
                mask = cv2.imdecode(nparr_mask, cv2.IMREAD_REDUCED_GRAYSCALE_8)
                mask = cv2.resize(mask, (256, 512))

                filename_result = "Result" + filename_image
                path_result = os.path.join(imagePath, filename_result)
                compute(image, image_path, mask, mask_path, path_result)
                inputImage = 'http://127.0.0.1:6000/results/' + filename_image
                predictedImage = 'http://127.0.0.1:6000/results/' + filename_result
                return render_template('HTML_result.html', entry=inputImage, prediction=predictedImage)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=6000,debug=True)
