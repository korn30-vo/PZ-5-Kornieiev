
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

train_X = train_X.reshape(-1, 28, 28, 1).astype("float32") / 255.
test_X = test_X.reshape(-1, 28, 28, 1).astype("float32") / 255.

train_Y_one = to_categorical(train_Y, 10)
test_Y_one = to_categorical(test_Y, 10)

model1 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(10, activation='softmax')
])

model1.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history1 = model1.fit(train_X, train_Y_one,
                      epochs=5,
                      batch_size=64,
                      validation_split=0.2)

model2 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.4),

    Dense(10, activation='softmax')
])

model2.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history2 = model2.fit(train_X, train_Y_one,
                      epochs=5,
                      batch_size=64,
                      validation_split=0.2)

model = model2

test_loss, test_acc = model.evaluate(test_X, test_Y_one)
print("Test accuracy:", test_acc)

predictions = model.predict(test_X)
pred_labels = np.argmax(predictions, axis=1)

print(classification_report(test_Y, pred_labels))

cm = confusion_matrix(test_Y, pred_labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

correct = np.where(pred_labels == test_Y)[0]
incorrect = np.where(pred_labels != test_Y)[0]

plt.figure(figsize=(10,5))


for i, idx in enumerate(correct[:3]):
    plt.subplot(2,3,i+1)
    plt.imshow(test_X[idx].reshape(28,28), cmap='gray')
    plt.title(f"OK: {pred_labels[idx]}")
    plt.axis('off')


for i, idx in enumerate(incorrect[:3]):
    plt.subplot(2,3,i+4)
    plt.imshow(test_X[idx].reshape(28,28), cmap='gray')
    plt.title(f"WRONG: {pred_labels[idx]}")
    plt.axis('off')

plt.show()


plt.figure()
plt.plot(history2.history['accuracy'], label='Train')
plt.plot(history2.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history2.history['loss'], label='Train')
plt.plot(history2.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()
plt.show()
