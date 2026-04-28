import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json

def train_blood_model(dataset_path, model_output_path):
    print(f"Starting training pipeline for {dataset_path}...")
    
    # Data Augmentation & Preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    
    # Save class indices immediately
    indices = {v: k for k, v in train_generator.class_indices.items()}
    with open(model_output_path.replace('.h5', '_classes.json'), 'w') as f:
        json.dump(indices, f)

    # Build Model using ResNet50V2 (More robust than MobileNet)
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for optimization
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        ModelCheckpoint(model_output_path, monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    # Stage 1: Train top layers
    print("Stage 1: Training top layers...")
    model.fit(
        train_generator, 
        epochs=12, 
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Stage 2: Fine-tuning
    print("Stage 2: Fine-tuning base model...")
    base_model.trainable = True
    # Freeze all layers except the last 50 for the larger blood dataset
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Very low LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator, 
        epochs=10, 
        validation_data=validation_generator,
        callbacks=callbacks
    )

    print(f"Training complete. Best model saved to {model_output_path}")

if __name__ == "__main__":
    dataset_path = "datasets/blood"
    model_output_path = "models/blood_model.h5"
    if os.path.exists(dataset_path):
        train_blood_model(dataset_path, model_output_path)
    else:
        print(f"Dataset path {dataset_path} not found.")
