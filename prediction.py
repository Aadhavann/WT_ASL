from xgboost import XGBClassifier
from visualize import get_landmarks, draw_landmarks_on_image
import pandas as pd
import numpy as np
import os

model = XGBClassifier()
model.load_model('model.json')

def preprocess(filename):
	landmarks = get_landmarks(filename)
	if len(landmarks) < 21:
		pass
	else:
		#data = np.array([landmarks[0].x, landmarks[1].x, landmarks[2].x, landmarks[3].x, landmarks[4].x, landmarks[5].x, landmarks[6].x, landmarks[7].x, landmarks[8].x, landmarks[9].x, landmarks[10].x, landmarks[11].x,landmarks[12].x, landmarks[13].x, landmarks[14].x, landmarks[15].x,landmarks[16].x, landmarks[17].x, landmarks[18].x, landmarks[19].x,landmarks[20].x,landmarks[0].y, landmarks[1].y, landmarks[2].y, landmarks[3].y,landmarks[4].y, landmarks[5].y, landmarks[6].y, landmarks[7].y,landmarks[8].y, landmarks[9].y, landmarks[10].y, landmarks[11].y,landmarks[12].y, landmarks[13].y, landmarks[14].y, landmarks[15].y,landmarks[16].y, landmarks[17].y, landmarks[18].y, landmarks[19].y,landmarks[20].y,landmarks[0].z, landmarks[1].z, landmarks[2].z, landmarks[3].z,landmarks[4].z, landmarks[5].z, landmarks[6].z, landmarks[7].z,landmarks[8].z, landmarks[9].z, landmarks[10].z, landmarks[11].z,landmarks[12].z, landmarks[13].z, landmarks[14].z, landmarks[15].z,landmarks[16].z, landmarks[17].z, landmarks[18].z, landmarks[19].z,landmarks[20].z])
		data = np.array([landmarks[8].y, landmarks[14].y, landmarks[1].x, landmarks[4].x, landmarks[11].z, landmarks[20].y, landmarks[1].z, landmarks[3].y, landmarks[17].z, landmarks[6].y, landmarks[7].y, landmarks[1].y,landmarks[2].x, landmarks[15].x, landmarks[17].y, landmarks[16].y,landmarks[20].x, landmarks[15].y, landmarks[18].y, landmarks[8].x,landmarks[9].z,landmarks[20].z, landmarks[3].z, landmarks[10].y, landmarks[14].x,landmarks[4].z, landmarks[12].x, landmarks[0].y, landmarks[2].y,landmarks[11].y, landmarks[8].x, landmarks[1].x, landmarks[2].z,landmarks[5].y, landmarks[4].y, landmarks[13].x, landmarks[9].y,landmarks[12].z, landmarks[19].z, landmarks[18].z, landmarks[16].z,landmarks[16].x,landmarks[12].y, landmarks[14].z, landmarks[10].z, landmarks[13].y,landmarks[7].z, landmarks[19].x, landmarks[5].x, landmarks[4].x,landmarks[7].x, landmarks[10].x, landmarks[17].x, landmarks[1].z,landmarks[9].x, landmarks[5].z, landmarks[18].x, landmarks[19].y,landmarks[6].z, landmarks[13].z, landmarks[11].x, landmarks[15].z,landmarks[8].z])
		data = data.reshape(1,-1)
		return data

def predict(file):
	df = preprocess(file)
	preds = model.predict(df)
	return preds

print('A-0 can be confused for E')
print(predict('images/A.jpg'))

print('B - 2')
print(predict('images/B.jpg'))

print('D - 4')
print(predict('images/D.jpg'))

print('I - 5')
print(predict('images/I.jpg'))

print('U - 21')
print(predict('images/U.jpg'))

