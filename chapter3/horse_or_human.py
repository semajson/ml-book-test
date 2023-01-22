from tensorflow.keras.preprocessing.image import ImageDataGenerator


# All images will be rescaled by 1./255
augmentationMax = True
augmentationMin = False
if augmentationMax:
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, # Rotating each image randomly up to 40 degrees left or right
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, # Randomly flipping the image horizontally or vertically
    fill_mode='nearest' # Filling in any missing pixels after a move or shear with nearest neighbors
  )
elif augmentationMin:
  # I think agumented too much, try a bit less:
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=40, # Rotating each image randomly up to 40 degrees left or right
    width_shift_range=0.2,
    height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, # Randomly flipping the image horizontally or vertically
    fill_mode='nearest' # Filling in any missing pixels after a move or shear with nearest neighbors
  )
else:
  # old with no augmentation
  train_datagen = ImageDataGenerator(rescale=1/255)

training_dir="horse-or-human/training"

train_generator = train_datagen.flow_from_directory(
  training_dir,
  target_size=(300, 300),
  class_mode='binary'
)



#validation
validation_dir = 'horse-or-human/validation/'
validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(
  validation_dir,
  target_size=(300, 300),
  # batch_size=128,
  class_mode='binary'
)

print(validation_generator)


import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu' ,
              input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
       optimizer=RMSprop(learning_rate=0.001),
       metrics=['accuracy'])


pre_train=True

if pre_train:
  from using_pre_trained_model import model_with_pre
  history = model_with_pre.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=40,
        verbose=1,
        validation_data=validation_generator)

  from test_horse_human import files, horse_or_human

  for file in files:
      horse_or_human(model_with_pre, file)

  test_file= "horse-or-human/validation/horses/horse1-000.png"
  horse_or_human(model_with_pre, test_file)

  test_file= "horse-or-human/validation/humans/valhuman01-00.png"
  horse_or_human(model_with_pre, test_file)

else:
  history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=20,
        verbose=1,
        validation_data=validation_generator)

# from horse_or_human import history, model