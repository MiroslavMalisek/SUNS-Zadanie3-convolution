import pathlib
import os
import PIL.Image
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomBrightness
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

batch_size = 32
img_height = 180
img_width = 180

# Zdroj: Seminar10
base_dir = 'images'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

animals_folders = list(pathlib.Path(train_dir).glob('*'))


# vytvorenie mnozin na prvu cast (analyza)

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



#----------------plot images + animals count--------------------------------------
#zdroj: https://kanoki.org/2021/05/11/show-images-in-grid-inside-jupyter-notebook-using-matplotlib-and-numpy/

def img_reshape(img):
    height = 110
    width = 110
    img = PIL.Image.open(img)
    img = img.resize((height, width))
    img = np.asarray(img)
    return img


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


#------------------------Imagenet analysis-----------------------------------------
def get_predictions_imagenet(model, dataset):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # iterate over the dataset
    for image_batch, label_batch in dataset:
        image_batch = tf.image.resize(image_batch, (260, 260))
        y_true.append(label_batch)
        preds = model.predict(image_batch, verbose=1)
        decoded_preds = decode_predictions(preds, top=1)
        predicted_classes = [pred[0][1] for pred in decoded_preds]
        y_pred.append(predicted_classes)

    class_labels = list(dataset.class_names)

    # convert the true and predicted labels into tensors
    y_true = tf.concat([item for item in y_true], axis=0)
    y_pred_classes = tf.concat([item for item in y_pred], axis=0)
    return y_true, y_pred_classes, class_labels


model = EfficientNetB2(weights='imagenet')
model.save('efficientNetB2.keras')
# model = load_model('efficientNetB2.keras')

# zdroj: seminar 11
# get predictions for train and test images (all images)
y_train_true, y_train_pred_classes, train_class_labels = get_predictions_imagenet(model, train_ds)
y_test_true, y_test_pred_classes, test_class_labels = get_predictions_imagenet(model, test_ds)
y_true_np = y_train_true.numpy()
# concatenate true values for train and test images
y_true_np = np.append(y_true_np, y_test_true.numpy())
# concatenate predicted values for train and test images
y_pred_np = y_train_pred_classes.numpy()
y_pred_np = np.append(y_pred_np, y_test_pred_classes.numpy())
# change numbers for string names of classes
y_true_classes = [train_class_labels[number] for number in y_true_np]
imagenet_preds_df = pd.DataFrame({'True class': y_true_classes, 'Predicted class': y_pred_np})

# zdroj: chatgpt
# get top 3 predictions for each class (animal)
counts = imagenet_preds_df.groupby(['True class', 'Predicted class']).size().reset_index(name='count')
counts.sort_values(['True class', 'count'], ascending=[True, False], inplace=True)
top_3_kinds = counts.groupby('True class').head(3)


# ------------------------------Druha cast - vytvorenie CNN--------------------------------------------

#zdroj: seminár 11
def get_predictions_train(model, dataset):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # iterate over the dataset
    for image_batch, label_batch in dataset:
        # append true labels
        y_true.append(np.argmax(label_batch, axis=-1))
        # compute predictions
        preds = model.predict(image_batch, verbose=0)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=-1))

    # convert the true and predicted labels into tensors
    y_true_classes = tf.concat([item for item in y_true], axis=0)
    y_pred_classes = tf.concat([item for item in y_pred], axis=0)

    return y_true_classes, y_pred_classes

def get_predictions_test(model, dataset):
    # get predictions from test data
    y_pred = model.predict(dataset)
    # convert predictions classes to one hot vectors
    y_pred_classes = np.argmax(y_pred, axis=1)
    # get true labels for test set
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    class_labels = list(dataset.class_names)

    return y_true, y_pred_classes, class_labels

#zdroj: ChatGPT
def plot_confusion_matrix(true_labels, predicted_labels, class_names, name):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    # Display the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=False, cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    matrix_name = "Confusion matrix for " + name + " dataset"
    plt.title(matrix_name)
    plt.tight_layout()
    plt.show()


train_ds_cnn = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds_cnn = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds_cnn = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=False,
    label_mode="categorical",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds_cnn.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds_cnn = train_ds_cnn.cache().prefetch(buffer_size=AUTOTUNE)
val_ds_cnn = val_ds_cnn.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

#zdroj: https://www.tensorflow.org/tutorials/images/classification
data_augmentation = Sequential([
    RandomFlip("horizontal",
               input_shape=(img_height,
                            img_width,
                            3)),
    RandomRotation(0.1)
])

#zdroj: https://pythonsimplified.com/image-classification-using-cnn-and-tensorflow-2/
model = Sequential([
    #data_augmentation,
    Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    #Rescaling(1./255),
    Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.1)),
    MaxPooling2D((3, 3)),
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l1(0.1)),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto', verbose=1)
history = model.fit(
    train_ds_cnn,
    validation_data=val_ds_cnn,
    epochs=15,
    callbacks=[early_stopping]
)

#Zdroj: zadanie1
train_scores = model.evaluate(train_ds_cnn, batch_size=batch_size, verbose=1)
test_scores = model.evaluate(test_ds_cnn, batch_size=batch_size, verbose=1)

print(f"Uspesnost (accuracy) na trenovacich datach: {train_scores[1]:.4f}")
print(f"Uspesnost (accuracy) na testovacich datach: {test_scores[1]:.4f}")

# Plot loss and accuracy
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Chyba v priebehu trenovania")
plt.xlabel("Epocha")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
# plt.savefig("KerasBestLoss.jpg")
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title("Uspesnost v priebehu trenovania")
plt.xlabel("Epocha")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
# plt.savefig("KerasBestAccuracy.jpg")
plt.show()

# functions to predict on batches
train_true_classes, train_pred_classes = get_predictions_train(model, train_ds_cnn)
test_true_classes, test_pred_classes = get_predictions_train(model, test_ds_cnn)
plot_confusion_matrix(train_true_classes.numpy(), train_pred_classes.numpy(), class_names, 'train')
plot_confusion_matrix(test_true_classes.numpy(), test_pred_classes.numpy(), class_names, 'test')

print()
