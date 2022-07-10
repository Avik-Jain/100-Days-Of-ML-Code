# Day 41 |Convolutional Neural Networks
## Importing the libraries
``
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
```
# Downloading the dataset
```
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
```
**The dataset contains five sub-directories, as per class:**
```
flower_photo/  
	daisy/  
	dandelion/  
	roses/  
	sunflowers/  
	tulips/
```
**There are 3,670 total images:**
```
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
```
```
3670      
```
# let's look some of them.
```
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
```
![rose image](https://www.tensorflow.org/static/tutorials/images/classification_files/output_N1loMlbYHeiJ_0.png)
```
PIL.Image.open(str(roses[1]))
```
![other rose](https://www.tensorflow.org/static/tutorials/images/classification_files/output_RQbZBOTLHiUP_0.png)
# Load data using a Keras
```
batch_size = 32
img_height = 180
img_width = 180
```
 **It's good practice to use a validation split when developing your model. 
Let's use 80% of the images for training, and 20% for validation.**
```
train_ds = tf.keras.utils.image_dataset_from_directory(  
	data_dir,  
	validation_split=0.2,  
	subset="training",  
	seed=123,  
	image_size=(img_height, img_width),  
	batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(  
	data_dir,  
	validation_split=0.2,  
	subset="validation",  
	seed=123,  
	image_size=(img_height, img_width),  
	batch_size=batch_size
)
```
```
Found 3670 files belonging to 5 classes.
Using 734 files for validation.
```
**class names is in the `class_names` attribute on these datasets**
```
class_names = train_ds.class_names
print(class_names)
```
```
['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
```
## Visualize some data
```
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):  
	for i in range(9):    
		ax = plt.subplot(3, 3, i + 1)    			
		plt.imshow(images[i].numpy().astype("uint8"))    	
		plt.title(class_names[labels[i]])    plt.axis("off")
```
![enter image description here](https://www.tensorflow.org/static/tutorials/images/classification_files/output_wBmEA9c0JYes_0.png)

## Configure the dataset for performance
**Make sure to use buffered prefetching so you can yield data from disk without having I/O become blocking**
```
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```
## Standardize the data

**The RGB channel values are in the  `[0, 255]`  range. This is not ideal for a neural network; in general you should seek to make your input values small.**
```
normalization_layer = layers.Rescaling(1./255)
```
```
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
```
# Create the model
```
num_classes = len(class_names)
model = Sequential([  
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), 
	layers.Conv2D(16, 3, padding='same', activation='relu'),  
	layers.MaxPooling2D(),  
	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),  
	layers.Conv2D(64, 3, padding='same', activation='relu'),  layers.MaxPooling2D(),  
	layers.Flatten(),  
	layers.Dense(128, activation='relu'),  
	layers.Dense(num_classes)
])
```
## Compile the model
```
model.compile(
	optimizer='adam',          
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),              
	metrics=['accuracy']
)
```
## Model summary

```
model.summary()
```
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling_1 (Rescaling)     (None, 180, 180, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 180, 180, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 90, 90, 16)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 90, 90, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 45, 45, 32)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 45, 45, 64)        18496     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 22, 22, 64)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 30976)             0         
                                                                 
 dense (Dense)               (None, 128)               3965056   
                                                                 
 dense_1 (Dense)             (None, 5)                 645       
                                                                 
=================================================================
Total params: 3,989,285
Trainable params: 3,989,285
Non-trainable params: 0
_________________________________________________________________
```

## Train the model
```
epochs=10
history = model.fit(  
	train_ds,  
	validation_data=val_ds,  
	epochs=epochs
)
```
```
Epoch 1/10
92/92 [==============================] - 3s 16ms/step - loss: 1.2769 - accuracy: 0.4489 - val_loss: 1.0457 - val_accuracy: 0.5804
Epoch 2/10
92/92 [==============================] - 1s 11ms/step - loss: 0.9386 - accuracy: 0.6328 - val_loss: 0.9665 - val_accuracy: 0.6158
Epoch 3/10
92/92 [==============================] - 1s 11ms/step - loss: 0.7390 - accuracy: 0.7200 - val_loss: 0.8768 - val_accuracy: 0.6540
Epoch 4/10
92/92 [==============================] - 1s 11ms/step - loss: 0.5649 - accuracy: 0.7963 - val_loss: 0.9258 - val_accuracy: 0.6540
Epoch 5/10
92/92 [==============================] - 1s 11ms/step - loss: 0.3662 - accuracy: 0.8733 - val_loss: 1.1734 - val_accuracy: 0.6267
Epoch 6/10
92/92 [==============================] - 1s 11ms/step - loss: 0.2169 - accuracy: 0.9343 - val_loss: 1.3728 - val_accuracy: 0.6499
Epoch 7/10
92/92 [==============================] - 1s 11ms/step - loss: 0.1191 - accuracy: 0.9629 - val_loss: 1.3791 - val_accuracy: 0.6471
Epoch 8/10
92/92 [==============================] - 1s 11ms/step - loss: 0.0497 - accuracy: 0.9871 - val_loss: 1.8002 - val_accuracy: 0.6390
Epoch 9/10
92/92 [==============================] - 1s 11ms/step - loss: 0.0372 - accuracy: 0.9922 - val_loss: 1.8545 - val_accuracy: 0.6390
Epoch 10/10
92/92 [==============================] - 1s 11ms/step - loss: 0.0715 - accuracy: 0.9813 - val_loss: 2.0656 - val_accuracy: 0.6049
```

## Visualize training results
```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
![enter image description here](https://www.tensorflow.org/static/tutorials/images/classification_files/output_jWnopEChMMCn_0.png)

## Overfitting
The training accuracy is increasing linearly over time, whereas validation accuracy stalls around 60% in the training process. Also, the difference in accuracy between training and validation accuracy is noticeable—a sign of overfitting
There are multiple ways to fight overfitting in the training process but i am using _Dropout_ to my modle.

## Dropout
technique to reduce overfitting is to introduce  dropout regularization to the network.
When you apply dropout to a layer, it randomly drops out (by setting the activation to zero) a number of output units from the layer during the training process. Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.
we can easily create it by tensorflow api tf.keras.layers.Dropout

```
model = Sequential([  
	layers.Rescaling(1./255),  
	layers.Conv2D(16, 3, padding='same', activation='relu'),  
	layers.MaxPooling2D(),  
	layers.Conv2D(32, 3, padding='same', activation='relu'), 
	layers.MaxPooling2D(),  
	layers.Conv2D(64, 3, padding='same', activation='relu'),  		  
	layers.MaxPooling2D(),  
	layers.Dropout(0.2),  
	layers.Flatten(),  
	layers.Dense(128, activation='relu'),  
	layers.Dense(num_classes)
])
```
```
model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),             
    metrics=['accuracy'])
```

```
epochs = 15
history = model.fit( 
	train_ds,  
	validation_data=val_ds,  
	epochs=epochs
)
```
```
Epoch 1/15
92/92 [==============================] - 2s 14ms/step - loss: 1.3840 - accuracy: 0.3999 - val_loss: 1.0967 - val_accuracy: 0.5518
Epoch 2/15
92/92 [==============================] - 1s 12ms/step - loss: 1.1152 - accuracy: 0.5395 - val_loss: 1.1123 - val_accuracy: 0.5545
Epoch 3/15
92/92 [==============================] - 1s 12ms/step - loss: 1.0049 - accuracy: 0.6052 - val_loss: 0.9544 - val_accuracy: 0.6253
Epoch 4/15
92/92 [==============================] - 1s 12ms/step - loss: 0.9452 - accuracy: 0.6257 - val_loss: 0.9681 - val_accuracy: 0.6213
Epoch 5/15
92/92 [==============================] - 1s 12ms/step - loss: 0.8804 - accuracy: 0.6591 - val_loss: 0.8450 - val_accuracy: 0.6798
Epoch 6/15
92/92 [==============================] - 1s 12ms/step - loss: 0.8001 - accuracy: 0.6945 - val_loss: 0.8715 - val_accuracy: 0.6594
Epoch 7/15
92/92 [==============================] - 1s 12ms/step - loss: 0.7736 - accuracy: 0.6965 - val_loss: 0.8059 - val_accuracy: 0.6935
Epoch 8/15
92/92 [==============================] - 1s 12ms/step - loss: 0.7477 - accuracy: 0.7078 - val_loss: 0.8292 - val_accuracy: 0.6812
Epoch 9/15
92/92 [==============================] - 1s 12ms/step - loss: 0.7053 - accuracy: 0.7251 - val_loss: 0.7743 - val_accuracy: 0.6989
Epoch 10/15
92/92 [==============================] - 1s 12ms/step - loss: 0.6884 - accuracy: 0.7340 - val_loss: 0.7867 - val_accuracy: 0.6907
Epoch 11/15
92/92 [==============================] - 1s 12ms/step - loss: 0.6536 - accuracy: 0.7469 - val_loss: 0.7732 - val_accuracy: 0.6785
Epoch 12/15
92/92 [==============================] - 1s 12ms/step - loss: 0.6456 - accuracy: 0.7500 - val_loss: 0.7801 - val_accuracy: 0.6907
Epoch 13/15
92/92 [==============================] - 1s 12ms/step - loss: 0.5941 - accuracy: 0.7735 - val_loss: 0.7185 - val_accuracy: 0.7330
Epoch 14/15
92/92 [==============================] - 1s 12ms/step - loss: 0.5824 - accuracy: 0.7735 - val_loss: 0.7282 - val_accuracy: 0.7357
Epoch 15/15
92/92 [==============================] - 1s 12ms/step - loss: 0.5771 - accuracy: 0.7851 - val_loss: 0.7308 - val_accuracy: 0.7343
```

## Visualize training results
```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
![enter image description here](https://www.tensorflow.org/static/tutorials/images/classification_files/output_dduoLfKsZVIA_0.png)

## Predict on new data
Finally, let's use our model to classify an image that wasn't included in the training or validation sets.

```
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
img = tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(    
"This image most likely belongs to {} with a {:.2f} percent confidence."    
.format(class_names[np.argmax(score)], 100 * np.max(score)))
```
```
Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg
122880/117948 [===============================] - 0s 0us/step
131072/117948 [=================================] - 0s 0us/step
This image most likely belongs to sunflowers with a 85.13 percent confidence.
```

