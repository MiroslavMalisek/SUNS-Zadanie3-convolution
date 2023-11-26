import pathlib
import os
import PIL.Image
import tensorflow as tf

batch_size = 32
img_height = 180
img_width = 180

base_dir = 'images'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

animals_folders = list(pathlib.Path(train_dir).glob('*'))

for animal_folder in animals_folders[:5]:
    animal_images = list(animal_folder.glob('*'))
    im = PIL.Image.open(str(animal_images[0]))
    im.show()

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  shuffle=True,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)