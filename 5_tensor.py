import tensorflow as tf
from tensorflow.keras import layers, models, datasets, regularizers, initializers

import matplotlib.pyplot as plt

# Step 1: Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0       # Normalize

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Step 2: Function to create model
def create_model(initializer=None, dropout_rate=0.0, l2_reg=0.0):
    model = models.Sequential()
    
    # Hidden Layer 1
    model.add(layers.Dense(
        512,
        activation='relu',
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(l2_reg),
        input_shape=(3072,)
    ))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    # Hidden Layer 2
    model.add(layers.Dense(
        256,
        activation='relu',
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(l2_reg)
    ))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    # Hidden Layer 3
    model.add(layers.Dense(
        128,
        activation='relu',
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(l2_reg)
    ))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    # Output Layer
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Step 3: Training configurations
configs = {
    "Baseline": {"initializer": "glorot_uniform", "dropout": 0.0, "l2": 0.0},
    "Xavier": {"initializer": initializers.GlorotNormal(), "dropout": 0.0, "l2": 0.0},
    "Kaiming": {"initializer": initializers.HeNormal(), "dropout": 0.0, "l2": 0.0},
    "Dropout": {"initializer": "glorot_uniform", "dropout": 0.3, "l2": 0.0},
    "L2_Regularization": {"initializer": "glorot_uniform", "dropout": 0.0, "l2": 1e-4}
}

results = {}

# Step 4: Train and evaluate models
for name, cfg in configs.items():
    print(f"\nTraining: {name}")
    model = create_model(initializer=cfg["initializer"], dropout_rate=cfg["dropout"], l2_reg=cfg["l2"])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=128,
                        validation_data=(x_test, y_test),
                        verbose=2)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    
    results[name] = history

# Step 5: Plot Accuracy and Loss
plt.figure(figsize=(15,10))

for i, metric in enumerate(['accuracy', 'loss']):
    plt.subplot(2,1,i+1)
    for name, history in results.items():
        plt.plot(history.history[metric], label=f"{name} Train")
        plt.plot(history.history[f"val_{metric}"], label=f"{name} Val")
    plt.title(metric.capitalize())
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()

plt.show()
