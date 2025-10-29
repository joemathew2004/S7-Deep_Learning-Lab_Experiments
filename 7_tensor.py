import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


def prepare_data(img_size=(224, 224)):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize and convert to RGB (3 channels)
    x_train = np.stack((x_train,) * 3, axis=-1) / 255.0
    x_test = np.stack((x_test,) * 3, axis=-1) / 255.0

    # Resize to match VGG19 input
    x_train = tf.image.resize(x_train, img_size).numpy()
    x_test = tf.image.resize(x_test, img_size).numpy()

    # Train/validation split
    val_split = 0.2
    split_idx = int((1 - val_split) * len(x_train))
    x_val, y_val = x_train[split_idx:], y_train[split_idx:]
    x_train, y_train = x_train[:split_idx], y_train[:split_idx]

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# Model Definitions
def build_vgg19_fixed(num_classes=10):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze all layers

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_vgg19_finetuned(num_classes=10):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze only first 3 blocks (~20 layers)
    for layer in base_model.layers[:15]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_baseline_cnn(num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Training Function
def train_model(model, x_train, y_train, x_val, y_val, lr=1e-3, epochs=8, batch_size=32):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    train_time = time.time() - start
    return history, train_time


def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    return test_acc * 100, y_pred, y_true


def plot_training(history_list, names):
    plt.figure(figsize=(12, 5))
    for h, name in zip(history_list, names):
        plt.plot(h.history['accuracy'], label=f'{name} Train')
        plt.plot(h.history['val_accuracy'], '--', label=f'{name} Val')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    print("=== MNIST Classification with VGG19 (TensorFlow) ===")

    # Data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data()

    models_dict = {
        "VGG19-Fixed": build_vgg19_fixed(),
        "VGG19-FineTuned": build_vgg19_finetuned(),
        "Baseline-CNN": build_baseline_cnn()
    }

    results = {}
    histories = []
    names = []

    for name, model in models_dict.items():
        print(f"\n--- Training {name} ---")
        lr = 1e-3 if "Fixed" in name else 1e-4
        history, train_time = train_model(model, x_train, y_train, x_val, y_val, lr=lr)
        test_acc, y_pred, y_true = evaluate_model(model, x_test, y_test)
        results[name] = (test_acc, train_time, y_true, y_pred)
        histories.append(history)
        names.append(name)
        print(f"{name} | Test Accuracy: {test_acc:.2f}% | Training Time: {train_time:.2f}s")

    # Plot Training Accuracy
    plot_training(histories, names)

    # Confusion Matrices
    for name, (acc, _, y_true, y_pred) in results.items():
        plot_confusion(y_true, y_pred, name)
        print(f"\nClassification Report for {name}:\n")
        print(classification_report(y_true, y_pred))

    # Comparison Plot
    accs = [results[n][0] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accs, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Comparison - Test Accuracy')
    plt.ylabel('Accuracy (%)')
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{a:.2f}%", ha='center', va='bottom')
    plt.show()

    print("\n=== Summary ===")
    for name, (acc, train_time, _, _) in results.items():
        print(f"{name}: Test Accuracy = {acc:.2f}%, Training Time = {train_time:.2f}s")

if __name__ == "__main__":
    main()
