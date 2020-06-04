# 0 = Cough, 1 = Not Cough   
#the code will return probability of two classes, the first probability is of cough and second one is of not_cough,
#here it sucessfully returns the output as not_cough for dogs barking sound with the probability of 99%

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_HEIGHT = 480
IMG_WIDTH = 640
test_datagen = ImageDataGenerator(rescale=1./255)
predict_dir = "predict_data/"
print(os.listdir(predict_dir))
test_generator = test_datagen.flow_from_directory(
        directory = 'predict_data',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary',
        shuffle=False)
model = tf.keras.models.load_model('model.h5')
print("classification: ",model.predict_generator(test_generator))    
