import pathlib
import os
import PIL.Image
import tensorflow as tf
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from keras.models import load_model



def img_reshape(img):
    height = 110
    width = 110
    img = PIL.Image.open(img)
    img = img.resize((height, width))
    img = np.asarray(img)
    return img

def get_predictions_train(model, dataset):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # iterate over the dataset
    for image_batch, label_batch in dataset:
        #image_batch = tf.image.resize(image_batch, (260, 260))
        #image_batch = tf.keras.applications.efficientnet.preprocess_input(image_batch)
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(image_batch, verbose=0)
        dec = decode_predictions(preds, top=1)
        #print('Predicted:', decode_predictions(preds, top=1))
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors
    y_true = tf.concat([item for item in y_true], axis=0)
    y_pred_classes = tf.concat([item for item in y_pred], axis=0)
    class_labels = list(dataset.class_names)

    return y_true, y_pred_classes, class_labels

def get_predictions_test(model, dataset):
    # get predictions from test data
    dataset = preprocess_input(dataset)
    y_pred = model.predict(dataset)
    # convert predictions classes to one hot vectors
    y_pred_classes = np.argmax(y_pred, axis=1)
    # get true labels for test set
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    class_labels = list(dataset.class_names)

    return y_true, y_pred_classes, class_labels


batch_size = 32
img_height = 260
img_width = 260

#Zdroj: Seminar10
base_dir = 'images'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

animals_folders = list(pathlib.Path(train_dir).glob('*'))

#Zdroj: Seminar11
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=False,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=False,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds_cache = train_ds.cache().prefetch(buffer_size=AUTOTUNE)


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

#------------------------Imagenet analysis-----------------------------------------
#model = EfficientNetB2(weights='imagenet')
#model.save('efficientNetB2.keras')
model = load_model('efficientNetB2.keras')
#for el, lab in train_ds:
#    np.squeeze(el, axis=0)
#    print()

y_true, y_pred_classes, class_labels = get_predictions_train(model, train_ds)
print()