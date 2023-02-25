import numpy as np
import tensorflow.keras.preprocessing.image
from tensorflow.keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Model Loaded Successfully")

def classify(img_file):
    img_name = img_file
    test_image = tensorflow.keras.preprocessing.image.load_img(img_name, target_size = (256, 256),grayscale=True)

    test_image = tensorflow.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes=["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
    result = classes[max_prob - 1]
    print(img_name,result)


import os
path = 'E:/Internship Program/Artificial Intelligence/Deep Learning/HandGestureDatasetDay13Code/Dataset/test/FOUR'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.png' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')
