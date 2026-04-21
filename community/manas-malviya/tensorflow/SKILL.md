| name        | tensorflow-quickstart |
|-------------|----------------------|
| description | TensorFlow skill — Covers neural networks, CNNs, RNNs, LSTM, GRU, GANs, VAEs, model training, data pipelines, and practical deep learning workflows with code examples. |

---

# TensorFlow

## What This Is

TensorFlow is an open-source deep learning framework by Google used to build, train, and deploy machine learning models.

Use this when you're working on:

* Neural networks (ANN)
* Image tasks (CNN)
* Sequential data (RNN, LSTM, GRU)
* Generative models (GANs, VAEs)
* Production ML pipelines

---

## Installation

```bash
pip install tensorflow
```

---

## Quick Start

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
```

---

## Core Concepts

### Tensors

```python
import tensorflow as tf

x = tf.constant([[1, 2], [3, 4]])
y = tf.matmul(x, x)
```

---

### Custom Model (Subclassing)

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.d1(x)
        return self.d2(x)
```

---

## Core Architectures

### ANN

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

### CNN

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

### RNN / LSTM

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

---

### GRU

```python
model = tf.keras.Sequential([
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

---

## Generative Models

### GAN (Minimal Idea)

```python
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid')
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

---

### VAE (Encoder)

```python
encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])
```

---

## Data Pipelines (Important)

```python
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```

---

## Training Tricks

### Callbacks

```python
early_stop = tf.keras.callbacks.EarlyStopping(patience=3)
checkpoint = tf.keras.callbacks.ModelCheckpoint("best.h5", save_best_only=True)
```

---

### Learning Rate Scheduling

```python
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
```

---

## Useful Patterns

### Save / Load

```python
model.save("model.h5")
model = tf.keras.models.load_model("model.h5")
```

---

### Evaluation

```python
model.evaluate(x_test, y_test)
```

---

### Prediction

```python
preds = model.predict(x_test)
```

---

## Performance Notes

* Use **GPU / CUDA** for training speed
* Use **tf.data** for large datasets
* Normalize inputs (very important)
* Use **batching + prefetching**
* Avoid very large batch sizes blindly

---

## When to Use What

* ANN → simple tabular data
* CNN → images
* LSTM / GRU → sequences (text, time series)
* GAN / VAE → generation tasks

---

## References

* https://www.tensorflow.org/tutorials
* https://www.tensorflow.org/guide
* https://keras.io/examples/
* https://www.tensorflow.org/api_docs
* https://developers.google.com/machine-learning/crash-course

---

## Extras (Optional Exploration)

* Transfer Learning (ResNet, MobileNet)
* NLP with TensorFlow (tokenization, embeddings)
* TensorBoard for visualization
* Distributed training (multi-GPU / TPU)
