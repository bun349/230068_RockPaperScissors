import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Preprocessing
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'd:/Code bisa di D/Matkul S4/AI/230068_RockPaperScissors/rock_paper_scissors/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'd:/Code bisa di D/Matkul S4/AI/230068_RockPaperScissors/rock_paper_scissors/val',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Model CNN
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(0.83, color='gray', linestyle='--', label='Target 83%')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print final accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Akurasi akhir pada data training: {final_train_acc:.4f}")
print(f"Akurasi akhir pada data validasi: {final_val_acc:.4f}")

# Load and predict on test images
test_images = [
    ('d:/Code bisa di D/Matkul S4/AI/230068_RockPaperScissors/test/dda.jpg', 'Paper'),
    ('d:/Code bisa di D/Matkul S4/AI/230068_RockPaperScissors/test/landscape-8306693_1280.jpg', 'Rock'),
    ('d:/Code bisa di D/Matkul S4/AI/230068_RockPaperScissors/test/sdi-scissors-5837-1.jpg', 'Scissors')
]

class_names = list(train_generator.class_indices.keys())
print("\nHasil Prediksi:")

for i, (path, true_label) in enumerate(test_images, start=1):
    img = keras.preprocessing.image.load_img(path, target_size=(150, 150))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    print(f"Gambar {i} (seharusnya {true_label}): Ditebak sebagai {predicted_class}")
