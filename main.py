import pathlib
import os
import PIL.Image
import tensorflow as tf
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


def img_reshape(img):
    height = 110
    width = 110
    img = PIL.Image.open(img)
    img = img.resize((height, width))
    img = np.asarray(img)
    return img


batch_size = 32
img_height = 180
img_width = 180

#Zdroj: Seminar10
base_dir = 'images'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

animals_folders = list(pathlib.Path(train_dir).glob('*'))

#Zdroj: Seminar11
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

"""
#----------------plot images + animals count--------------------------------------
#zdroj: https://kanoki.org/2021/05/11/show-images-in-grid-inside-jupyter-notebook-using-matplotlib-and-numpy/
img_arr_to_show = []
for animal_folder in animals_folders:
    img_arr_to_show.append(img_reshape(str(list(animal_folder.glob('*'))[0])))

fig = plt.figure(figsize=(18, 12))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(9, 10),# creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes
                 )

for ax, im, name in zip(grid, img_arr_to_show, class_names):
    ax.axis('off')
    ax.set_title(name)
    ax.imshow(im)

plt.tight_layout()
plt.savefig("representants.jpg")
plt.show()
"""
"""
#Zdroj: https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-list-of-dicts/ + ChatGPT
animals_count_list = []
for animal_name in class_names:
    train_animal_images = list(pathlib.Path(os.path.join(base_dir, 'train', animal_name)).glob('*'))
    test_animal_images = list(pathlib.Path(os.path.join(base_dir, 'test', animal_name)).glob('*'))
    animals_count_list.append({'Animal': animal_name, "Train_count": len(train_animal_images),
                               "Test_count": len(test_animal_images),
                               "Total_count": (len(train_animal_images) + len(test_animal_images))})

animals_count_df = pd.DataFrame(animals_count_list, columns=['Animal', 'Train_count', 'Test_count', 'Total_count'])
"""



print()