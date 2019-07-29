import os 
from keras import layers 
from keras import models
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


#input_dir = '/Users/samanvay/Downloads/statoil-iceberg-classifier-challenge'
input_dir = os.getcwd()

train = pd.read_json('{0}/train/processed/train.json'.format(input_dir))

train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

images = []
angles = []

for i, row in train.iterrows():
    # Convert the flattened bands as 75x75 arrays
    band_1 = np.array(row['band_1']).reshape(75, 75)
    band_2 = np.array(row['band_2']).reshape(75, 75)

    # Rescale
    ver = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
    hor = (band_2 - band_2.min()) / (band_2.max() - band_2.min())

    two_band = np.dstack((ver,hor))
    images.append(two_band)


    inc_angle = row['inc_angle']
    angles.append(inc_angle) 

X = np.array(images)
angles = np.array(angles)
y = to_categorical(train.is_iceberg.values,num_classes=2)

img_tr, img_val, y_tr, y_val = train_test_split(X, y, shuffle=False, test_size=0.20)
ang_tr, ang_val, y_tr, y_val = train_test_split(angles, y, shuffle=False, test_size=0.20)


main_input = layers.Input(shape=(75,75,2))

img_conv1 = layers.Conv2D(32, kernel_size=3, activation='relu')(main_input)
img_pool1 = layers.MaxPooling2D(pool_size=(2, 2))(img_conv1)

img_conv2 = layers.Conv2D(64, kernel_size=3, activation='relu')(img_pool1)
img_pool2 = layers.MaxPooling2D(pool_size=(2, 2))(img_conv2)

img_conv3 = layers.Conv2D(64, kernel_size=3, activation='relu')(img_pool2)
img_pool3 = layers.MaxPooling2D(pool_size=(2, 2))(img_conv3)

img_conv4 = layers.Conv2D(64, kernel_size=3, activation='relu')(img_pool3)
img_pool4 = layers.MaxPooling2D(pool_size=(2, 2))(img_conv4)

img_flat = layers.Flatten()(img_pool4)

auxiliary_input = layers.Input(shape=(1,))
merged = layers.concatenate([img_flat, auxiliary_input])

dense1 = layers.Dense(64, activation='relu')(merged)
main_output = layers.Dense(2, activation='sigmoid')(dense1)

model = models.Model(inputs=[main_input,auxiliary_input],outputs=main_output)

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit([img_tr,ang_tr],y_tr, batch_size=32, epochs=100)
test_loss, test_acc = model.evaluate([img_val,ang_val], y_val)

print(test_acc)

###########################################
# SUBMISSION
###########################################


test = pd.read_json('{0}/test/processed/test.json'.format(input_dir))

#Xtest = get_images(test)

test_images = []
test_angles = []

for i, row in test.iterrows():
    # Convert the flattened bands as 75x75 arrays
    band_1 = np.array(row['band_1']).reshape(75, 75)
    band_2 = np.array(row['band_2']).reshape(75, 75)

    # Rescale
    ver = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
    hor = (band_2 - band_2.min()) / (band_2.max() - band_2.min())

    two_band = np.dstack((ver,hor))
    test_images.append(two_band)


    inc_angle = row['inc_angle']
    test_angles.append(inc_angle) 

test_img = np.array(test_images)
test_ang = np.array(test_angles)

test_predictions = model.predict([test_img,test_ang])

print(test_predictions)

print(test_predictions.shape)

submission = pd.DataFrame({'id': test['id'], 'is_iceberg': test_predictions[:, 1]})
submission.to_csv('sub_fcn.csv', index=False)
