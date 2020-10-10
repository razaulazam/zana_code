import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from sklearn.utils import class_weight

print("Eager Mode: ", tf.executing_eagerly())

data_set = pd.read_excel('SampleDataset.xlsx', sheet_name='Sheet1')

"""------- Data Preparation -------"""
train_data_labels = data_set['Label']
train_data = np.array(data_set.drop('Label', axis=1))

enc_label = LabelEncoder().fit(train_data_labels)
train_data_labels = enc_label.transform(train_data_labels)

_, ds_test, _, ds_test_labels = train_test_split(train_data, train_data_labels, test_size=0.4, random_state=42)

# Preparing the class weights since one of the classess is imbalanced i.e. Prognosis
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_data_labels), train_data_labels)
class_weights_dict = dict(enumerate(class_weights))

print(f"Number of training examples are: {len(train_data)}")
print(f"Number of testing examples are: {len(ds_test)}")

"""--------------------------------"""

"""------ Defining the model ------"""
# Using the transfer learning for the embedding vectors for the sentences.
# We could use another model but I picked up the smaller one which produces a 20 dimensional output vector.
model_hub = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(model_hub, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='softmax'))


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

"""---------------------------------"""

"""--- Training and testing the model ---"""
history = model.fit(train_data,
                    train_data_labels,
                    epochs=60,
                    batch_size=4,
                    verbose=1, class_weight=class_weights_dict)

results = model.evaluate(ds_test, ds_test_labels)
# ~83 % test accuracy is achieved. Could be improved by using a bigger and a better model.
"""-----------------------------"""








