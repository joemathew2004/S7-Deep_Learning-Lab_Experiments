import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.applications import VGG19
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Convert grayscale (28x28) → RGB (32x32x3) for VGG19
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_train = tf.image.grayscale_to_rgb(tf.image.resize(x_train, [32, 32]))
x_test = tf.image.grayscale_to_rgb(tf.image.resize(x_test, [32, 32]))

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Baseline CNN (for comparison)
baseline = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

baseline.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining Baseline CNN...")
baseline.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)

baseline_acc = baseline.evaluate(x_test, y_test, verbose=0)[1]
print(f"Baseline CNN Test Accuracy: {baseline_acc * 100:.2f}%")

#-------------------------------------------------------------------------------------------

# Pre-trained VGG19 as Feature Extractor
vgg_base = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg_base.trainable = False          # Freeze VGG19 weights

# Build new model
vgg_model = models.Sequential([
    vgg_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining Transfer Learning (VGG19 Feature Extractor)...")
vgg_model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)

vgg_acc = vgg_model.evaluate(x_test, y_test, verbose=0)[1]
print(f"VGG19 Feature Extractor Test Accuracy: {vgg_acc * 100:.2f}%")

#-------------------------------------------------------------------------------------------

# Fine-Tuning (Unfreeze top VGG layers)
vgg_base.trainable = True
for layer in vgg_base.layers[:-4]:  # Freeze most layers, train top 4
    layer.trainable = False

vgg_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

print("\nFine-tuning VGG19 (last few layers)...")
vgg_model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=1)

fine_tuned_acc = vgg_model.evaluate(x_test, y_test, verbose=0)[1]
print(f"Fine-tuned VGG19 Test Accuracy: {fine_tuned_acc * 100:.2f}%")

# Visualization
accs = [baseline_acc, vgg_acc, fine_tuned_acc]
labels = ['Baseline CNN', 'VGG19 (Feature Extractor)', 'VGG19 (Fine-tuned)']

plt.figure(figsize=(8,5))
plt.bar(labels, [a*100 for a in accs], color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Performance Comparison: MNIST Classification')
plt.ylabel('Test Accuracy (%)')
plt.ylim(95, 100)
plt.show()

# Predictions Visualization
preds = vgg_model.predict(x_test[:10])
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(y_test[:10], axis=1)

plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {pred_classes[i]}, True: {true_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
