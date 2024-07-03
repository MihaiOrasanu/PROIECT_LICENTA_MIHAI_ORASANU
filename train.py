import numpy as np
import librosa.display
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow as tf

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)


def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        if file.endswith('.wav'):
            input_file = os.path.join(input_path, file)
            output_file = os.path.join(output_path, file.replace('.wav', '.png'))
            create_spectrogram(input_file, output_file)


create_pngs_from_wavs('mixare/engine_idling_voce', 'Spectrograms/engine_idling_voce')
create_pngs_from_wavs('mixare/engine_idling', 'Spectrograms/engine_idling')
create_pngs_from_wavs('mixare/siren_voce', 'Spectrograms/siren_voce')
create_pngs_from_wavs('mixare/siren', 'Spectrograms/siren')
create_pngs_from_wavs('mixare/jackhammer_voce', 'Spectrograms/jackhammer_voce')
create_pngs_from_wavs('mixare/jackhammer', 'Spectrograms/jackhammer')


def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        if file.endswith('.png'):
            images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
            labels.append(label)

    return images, labels


def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)
    plt.show()


x = []
y = []

images, labels = load_images_from_path('Spectrograms/engine_idling_voce', 0)
show_images(images)
x += images
y += labels

images, labels = load_images_from_path('Spectrograms/engine_idling', 1)
show_images(images)
x += images
y += labels

images, labels = load_images_from_path('Spectrograms/siren_voce', 2)
show_images(images)
x += images
y += labels

images, labels = load_images_from_path('Spectrograms/siren', 3)
show_images(images)
x += images
y += labels

images, labels = load_images_from_path('Spectrograms/jackhammer_voce', 4)
show_images(images)
x += images
y += labels

images, labels = load_images_from_path('Spectrograms/jackhammer', 5)
show_images(images)
x += images
y += labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x_train_norm = preprocess_input(np.array(x_train))
x_test_norm = preprocess_input(np.array(x_test))

train_features = base_model.predict(x_train_norm)
test_features = base_model.predict(x_test_norm)

model = Sequential()
model.add(Flatten(input_shape=train_features.shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dense(6, activation='softmax'))  # Change to 6 as there are 6 classes
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(train_features, y_train_encoded, validation_data=(test_features, y_test_encoded), batch_size=8, epochs=7)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Acuratețe antrenare')
plt.plot(epochs, val_acc, ':', label='Acuratețe validare')
plt.title('Graficul antrenării rețelei MobileNet')
plt.xlabel('Epoca')
plt.ylabel('Acuratețea')
plt.legend(loc='lower right')
plt.show()

sns.set()

y_predicted = model.predict(test_features)
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
class_labels = ['engine_idling_voce', 'engine_idling', 'siren_voce', 'siren', 'jackhammer_voce', 'jackhammer']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Clasa prezisă')
plt.ylabel('Clasa reală')
plt.show()

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model salvat ca model.tflite")