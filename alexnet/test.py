from keras.models import Model, load_model
import cv2
from keras.preprocessing import image
import numpy as np

model1_path = "epoch_3dcnn_84.hdf5"
classifier = load_model('epoch_3dcnn_84.hdf5')
img_row, img_height, img_depth = 32,32,3


img = image.load_img("Test_dataset/Cabbage/C.jpg",target_size=(32,32))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)
classes = ["Broccoli", "Cabbage", "Capsicums", "Carrots", "Cauliflower", "Celeriac", "Celery", "Chilli peppers", "Chokos", "Courgettes and Scallopini", "Cucumber", "Eggplant", "Fennel", "Fresh  garnishes and flowers", "Garlic", "Ginger", "Indian vegetables", "Kale and Cavolo Nero", "Kohlrabi", "Melons", "Mushrooms", "Okra", "Onions", "Potatoes", "Pumpkins", "Radishes", "Spinach", "Spring onions", "Sweet corn", "Tomatoes", "Turnips", "Yams"]
classes= { i : classes[i] for i in range(0, len(classes) ) }
print(classes)
output = classifier.predict(img)
print(output)
output = np.argmax(output,axis = 1)
print(int(output[0]))
print(classes[int(output[0])])
