import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
#get data
(training_images, training_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
#preprocess
training_images = training_images / 255
test_images = test_images / 255
training_labels = keras.utils.to_categorical(training_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)
#Construct model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
#run
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
history=model.fit(training_images, training_labels,validation_data=(test_images, test_labels), epochs=15, callbacks=[early_stopping])
#show model
model.summary()
tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
)
# get plot
import matplotlib.pyplot as plt
def summarize_diagnostics(history):
  acc = history.history['accuracy']
  loss = history.history['loss']
  val_acc = history.history['val_accuracy']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  plt.plot(acc, 'r', label='Training accuracy')
  plt.plot(val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend(loc=0)
  plt.figure()
  plt.show()

  plt.plot(loss, 'r', label='Training Loss function')
  plt.plot(val_loss, 'b', label='Validation Loss function')
  plt.title('Training and validation loss function')
  plt.legend(loc=0)
  plt.figure()
  plt.show()
summarize_diagnostics(history)
import numpy as np
import seaborn as sn
# Make predictions on the test data
predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_labels, axis=1)
confusion_matrix = tf.math.confusion_matrix(test_labels, predictions)
f, ax = plt.subplots(figsize=(9, 7))
sn.heatmap(
    confusion_matrix,
    annot=True,
    linewidths=.5,
    fmt="d",
    square=True,
    ax=ax
)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
import math
numbers_to_display = 64
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10, 10))
random_num=np.random.randint(0,10000, size=numbers_to_display)
for plot_index in range(numbers_to_display):    
    predicted_label = predictions[random_num[plot_index]]
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    color_map = 'Greens' if predicted_label == test_labels[random_num[plot_index]] else 'Reds'
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(test_images[random_num[plot_index]].reshape((28, 28)), cmap=color_map)
    plt.xlabel(predicted_label)

plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show()
