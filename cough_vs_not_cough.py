#binary cnn classifier, 0 - cough, 1 - not_cough
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator()
train_dr = 'img_data/'

train_cough = os.path.join(train_dr,'0')
train_not_cough = os.path.join(train_dr,'1')

num_cough_tr = len(os.listdir(train_cough))
num_not_cough_tr = len(os.listdir(train_not_cough))

total_train = num_cough_tr + num_not_cough_tr
print('total training images: ',total_train)

batch_size = 50
epochs = 10
IMG_HEIGHT = 480
IMG_WIDTH = 640

train_image_generator = ImageDataGenerator(rescale = 1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
	directory = train_dr,
	shuffle = True,
	target_size = (IMG_HEIGHT, IMG_WIDTH),
	class_mode = 'binary')


model = Sequential()
model.add(Conv2D(input_shape=(IMG_HEIGHT,IMG_WIDTH,3), filters=2, kernel_size=4, strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs
)

#classifier = model.fit_generator(train, epochs=100)

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

model.save('model.h5')

print("classification: ",model.predict_generator(test_generator))    
