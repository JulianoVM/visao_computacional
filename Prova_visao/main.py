import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import kagglehub
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Baixando o dataset
dataset_dir = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")

# Carregando dataset com TensorFlow
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,  # Sem /PetImages
    labels='inferred',
    label_mode='int',
    image_size=(256, 256),
    batch_size=32,
    shuffle=True,
    seed=123
)

# Dividindo dataset
dataset = dataset.shuffle(1000, seed=123)

train_size = int(0.8 * tf.data.experimental.cardinality(dataset).numpy())
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Normalizando
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Criando a CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compilando
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinando
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

# Avaliando no test_dataset
y_true = []
y_pred = []

for images_batch, labels_batch in test_dataset:
    preds = model.predict(images_batch)
    preds = np.argmax(preds, axis=1)
    y_pred.extend(preds)
    y_true.extend(labels_batch.numpy())

# Métricas
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Plot comparativo
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.title('Comparação Treinamento vs Validação')
plt.show()

# --- Avaliação nas 6 imagens externas ---
print("\n### Avaliando nas 6 imagens externas da pasta './imagens' ###")

class_names = dataset.class_names  # Deve ser ['Cat', 'Dog']

image_dir = './imagens'
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

plt.figure(figsize=(12, 8))

for i, img_path in enumerate(image_paths):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)
    img_array = img_array / 255.0  # Normalização

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]

    plt.subplot(2, 3, i+1)
    plt.imshow(img.astype(np.uint8))
    plt.title(f"Pred: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
