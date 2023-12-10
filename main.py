import pathlib
import os
import PIL.Image
import tensorflow as tf
import pandas as pd
import math
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from keras.models import load_model, Model
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, RandomBrightness
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import SilhouetteVisualizer

batch_size = 32
img_height = 180
img_width = 180

# Zdroj: Seminar10
base_dir = 'images'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

animals_folders = list(pathlib.Path(train_dir).glob('*'))

"""
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
"""

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


"""
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

"""

"""
# ----------------------------tretia cast-----------------------------------------------------------------------
# ----------------------------priznaky do dataframe-------------------------------------------------------------
#Zdroj: Seminar11
train_ds_imagenet = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=False,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds_imagenet = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=False,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds_imagenet.class_names
train_files_paths = train_ds_imagenet.file_paths
test_files_paths = test_ds_imagenet.file_paths

AUTOTUNE = tf.data.AUTOTUNE
train_ds_imagenet = train_ds_imagenet.cache().prefetch(buffer_size=AUTOTUNE)
test_ds_imagenet = test_ds_imagenet.cache().prefetch(buffer_size=AUTOTUNE)


def extract_features(model, dataset):
    # get predictions from test data
    features = model.predict(dataset)
    global_average_layer = GlobalAveragePooling2D()
    return global_average_layer(features).numpy()


# model2 = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
# model2.save('efficientNetB2_false.keras')
model2 = load_model('efficientNetB2_false.keras')

"""

# zdroj: seminar 11
# get predictions for train and test images (all images)
"""
train_features_df = pd.DataFrame(extract_features(model2, train_ds_imagenet))
train_features_df['file_path'] = train_files_paths
train_features_df['class'] = np.concatenate([y for x, y in train_ds_imagenet], axis=0)

test_features_df = pd.DataFrame(extract_features(model2, test_ds_imagenet))
test_features_df['file_path'] = test_files_paths
test_features_df['class'] = np.concatenate([y for x, y in test_ds_imagenet], axis=0)

features_df = pd.concat([train_features_df, test_features_df], ignore_index=True)
features_df = features_df.sample(frac=1).reset_index(drop=True)
features_df.to_csv('features_df.csv')
"""

"""
features_df = pd.read_csv('features_df.csv')
features_df.drop(features_df.columns[0], axis=1, inplace=True)
#input and output features
features_df_X = features_df.drop(columns=['file_path', 'class'])
features_df_y = features_df[['file_path', 'class']]
"""

"""
#PCA redukcia
#zdroj Zadanie 2
variance = 0.95
pca = PCA(n_components=variance)
features_df_X_pca = pca.fit_transform(features_df_X)
number_pca = pca.n_components_
features_df_X_pca = pd.DataFrame(data=features_df_X_pca, columns=[f'PC{i}' for i in range(0, number_pca)])
"""

"""
#find optimal number of k using elbow method
#zdroj: ChatGPT
k_values = range(1, 25)
inertia = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_df_X)
    inertia.append(kmeans.inertia_)
# Plot the elbow curve
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method to Find Optimal k')
plt.show()


#DBSCAN optima parameters
#zdroj: https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd
neighbors = NearestNeighbors(n_neighbors=100)
neighbors_fit = neighbors.fit(features_df_X)
distances, indices = neighbors_fit.kneighbors(features_df_X)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.title('DBSCAN parameters estimation')
plt.plot(distances)
plt.show()

#find optimal number of k using silhouette method
# zdroj: https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html
fig, ax = plt.subplots(5, 2, figsize=(15,20))
visualizer = None
for i in range(2, 12):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(features_df_X)

visualizer.show()
plt.show()
"""
"""
#---------------kmeans--------------------
# zdroj: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
kmeans = KMeans(n_clusters=11, random_state=42).fit(features_df_X)
kmean_labels = kmeans.labels_
features_df['kmeans_cluster'] = kmean_labels
# features_df.to_csv('features_df_kmeans.csv')
"""

"""
def kmean_img_open_reshape(img):
    height = 150
    width = 150
    img = PIL.Image.open(img)
    img = img.resize((height, width))
    return img

# zdroj: ChatGPT
def show_images_for_kmean(df, kmean_value):
    fig = plt.figure(figsize=(18, 12))
    plt.title(f'Images for kmean cluster {kmean_value}', fontsize=30)

    # Filter DataFrame for the given kmean value
    kmean_df = df[df['kmeans_cluster'] == kmean_value]
    kmean_df.reset_index(drop=True, inplace=True)
    if kmean_value == 1:
        number_cols = 10
    elif kmean_value == 8:
        number_cols = 7
    else:
        number_cols = 5
    number_rows = math.ceil(len(kmean_df)/number_cols)
    # Plot images in a grid
    for index, row in kmean_df.iterrows():
        path = row['file_path']
        image = kmean_img_open_reshape(path)

        plt.subplot(number_rows, number_cols, index + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.axis('off')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def kmean_open_img_asarray(img):
    height = 300
    width = 300
    img = PIL.Image.open(img)
    img = img.resize((height, width))
    #img = np.asarray(img)
    return img


def show_avg_image_kmean(df, kmean_value):
    images_np = []
    # Filter DataFrame for the given kmean value
    kmean_df = df[df['kmeans_cluster'] == kmean_value]
    kmean_df.reset_index(drop=True, inplace=True)

    for index, row in kmean_df.iterrows():
        path = row['file_path']
        images_np.append(kmean_open_img_asarray(path))

    images_np = np.array(images_np)
    mean_image = np.mean(images_np, axis=0).astype(np.uint8)
    image = PIL.Image.fromarray(mean_image)
    image.save(f'Avg_image_kmean{kmean_value}.jpg')
    image.show()


features_df = pd.read_csv('features_df_kmeans.csv')
features_df.drop(features_df.columns[0], axis=1, inplace=True)
# features_df.sort_values(by=['kmeans_cluster', 'class'], inplace=True)
# zdroj: ChatGPT
kmeans_unique_class_df = features_df.groupby('kmeans_cluster').apply(lambda group: group.groupby('class').first())
kmeans_unique_class_df.reset_index(drop=True, inplace=True)
kmeans_unique_class_df = kmeans_unique_class_df[['file_path', 'kmeans_cluster']]

# -----------------------------kmeans images-------------------------------------
# Display grids for each unique kmean value
unique_kmeans = kmeans_unique_class_df['kmeans_cluster'].unique()
for kmean_value in unique_kmeans:
    show_images_for_kmean(kmeans_unique_class_df, kmean_value)

#average image
for kmean_value in unique_kmeans:
    # show_avg_image_kmean(features_df[['file_path', 'kmeans_cluster']], kmean_value)
    show_avg_image_kmean(kmeans_unique_class_df, kmean_value)
"""


# ---------------------------transfer learning on imagenet-----------------------------------------
train_ds_for_transfer = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds_for_transfer = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds_for_transfer = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=False,
    label_mode="categorical",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds_for_transfer.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds_for_transfer = train_ds_for_transfer.cache().prefetch(buffer_size=AUTOTUNE)
val_ds_for_transfer = val_ds_for_transfer.cache().prefetch(buffer_size=AUTOTUNE)
test_ds_for_transfer = test_ds_for_transfer.cache().prefetch(buffer_size=AUTOTUNE)

model3 = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
# model3.save('efficientNetB2_false.keras')
# model3 = load_model('efficientNetB2_false.keras')
model3.trainable = False
"""
model3.trainable = False
inputs = tf.keras.Input(shape=(img_width, img_width, 3))
x = preprocess_input(inputs)
x = model3(x, training=False)
x = GlobalAveragePooling2D()(x)
hidden = Dense(1024, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(hidden)
model3_final = Model(inputs, outputs)
"""


output = model3.output
# Condense feature maps from the output
output = GlobalAveragePooling2D()(output)
output = Dropout(0.2)(output)
# Add dense fully connected artificial neural network at the end
output = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(output)
## Final layer has 2 output neurons since we're classifying beds and sofas
final_output = Dense(num_classes, activation='softmax')(output)
model3_final = Model(inputs=model3.input, outputs=final_output)


model3_final.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto', verbose=1)
history = model3_final.fit(
    train_ds_for_transfer,
    validation_data=val_ds_for_transfer,
    epochs=15,
    callbacks=[early_stopping]
)

train_scores = model3_final.evaluate(train_ds_for_transfer, batch_size=batch_size, verbose=1)
test_scores = model3_final.evaluate(test_ds_for_transfer, batch_size=batch_size, verbose=1)

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
train_true_classes, train_pred_classes = get_predictions_train(model3_final, train_ds_for_transfer)
test_true_classes, test_pred_classes = get_predictions_train(model3_final, test_ds_for_transfer)
plot_confusion_matrix(train_true_classes.numpy(), train_pred_classes.numpy(), class_names, 'train')
plot_confusion_matrix(test_true_classes.numpy(), test_pred_classes.numpy(), class_names, 'test')
print()
