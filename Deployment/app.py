from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.models as models
from PIL import Image

app = Flask(__name__)


vgg=models.vgg16_bn(pretrained=True)

for param in vgg.parameters():
    param.requires_grad=False
    
num_classes = 6
final_in_features = vgg.classifier[6].in_features
vgg.classifier[6] = nn.Linear(final_in_features, num_classes)

path='models/plant_disease_model_2.pt'
vgg.load_state_dict(torch.load(path))
vgg.eval()

data=[['Your Apple plant is affected by scab disease '],['Your Apple plant is healthy'],
	  ['Your Bell Pepper plant is affected by Bacterial spot disease'],['Your Bell Pepper plant is healthy'],
	  ['Your Tomato plant is affected by Early Blight disease'],['Your Tomato Plant is healthy']
	]
df=pd.DataFrame(data,columns=['disease_name'])

def model_predict(img_path,model):
	img=Image.open(img_path)
	image = img.resize((224, 224))
	input_data = TF.to_tensor(image)
	input_data = input_data.view((-1, 3, 224, 224))
	output = vgg(input_data)
	output = output.detach().numpy()
	index = np.argmax(output)
	# print("Original : ", img[64:-4])
	pred_csv = df["disease_name"][index]
	#print(pred_csv)
	return pred_csv
	
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
	
@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)
		preds = model_predict(file_path, vgg)
		return preds
		#return render_template('index.html')
	return None
	
	
	
	
if __name__ == '__main__':
    app.run(debug=True)