import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template , request
import matplotlib.image as matimg
UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save out image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) # save the image into upload folder
        # get predcitions
        pred_image , prediction = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
        # print(prediction)

        # generate report
        report = []

        for i , obj in enumerate(prediction):
            gray_image = obj['roi'] # gray scale image (array)
            eigen_image = obj['eig_img'].reshape(100,100) # eigen image (array)
            gender_name = obj['prediction_name'] # name
            score = round(obj['score']*100,2) # score

            # save gray scale and eigen in predict folder
            gray_img_name = f'roi_{i}.jpg'
            eig_img_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_img_name}',gray_image,cmap='gray')
            matimg.imsave(f'./static/predict/{eig_img_name}',eigen_image,cmap='gray')

            # save report
            report.append([gray_img_name,
                           eig_img_name,
                           gender_name,score])
        return render_template('genderapp.html',fileupload =True, report = report) # post request
        

    return render_template('genderapp.html', fileupload = False) # get request