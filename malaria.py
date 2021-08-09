import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix


folder_parasitized = 'C:\\Users\\keert\\Datasets\\cell_images\\train\\parasitized'
folder_uninfected = 'C:\\Users\\keert\\Datasets\\cell_images\\train\\uninfected'

# checking how many images do we have for each class

print(len(os.listdir(folder_parasitized)))
print(len(os.listdir(folder_uninfected)))


def visualize_cell(cell='p', n=5):
    """
    mount a simple visualization of cell images
    """
    p_cells = os.listdir(folder_parasitized)
    u_cells = os.listdir(folder_uninfected)

    if cell == 'p':
        folder = 'Parasitized/' 
        cells = p_cells
    else:
        folder = 'Uninfected/'
        cells = u_cells
    
    plt.figure(figsize=(10,8))
    for i in range(n):
        plt.subplot(1,n,i+1)
        img=imread('C:\\Users\\keert\\Datasets\\cell_images\\train/' + folder + cells[i])
        plt.title(cell + ' ' +  str(img.shape))
        plt.imshow(img)
        plt.tight_layout()
    plt.show()
    

    
    print(visualize_cell(cell='p', n=5))
    print(visualize_cell(cell='u', n=5))



def check_dimensions(cell='p'):
    """
    plot a jointplot showing dimensions distributions of cell images
    """
    p_cells = [i for i in os.listdir(folder_parasitized) if i.endswith(".png")]
    u_cells = [i for i in os.listdir(folder_uninfected) if i.endswith(".png")]

    if cell=='p':
        folder = 'Parasitized/'
        cells = p_cells
    else:
        folder = 'Uninfected/'
        cells = u_cells
        
    dim1 = []
    dim2 = []

    for image_file in cells:
        img = imread('C:\\Users\\keert\\Datasets\\cell_images\\train/' + folder +  image_file)
        d1,d2,colors = img.shape
        dim1.append(d1)
        dim2.append(d2)
        
    print(np.mean(dim1))
    print(np.mean(dim2))
        
    sns.jointplot(x=dim1,y=dim2)





check_dimensions(cell='p')




check_dimensions(cell='u')

image_shape = (130,130,3)


train_data_dir = 'C:\\Users\\keert\\Datasets\\cell_images\\train/'
batch_size = 64

image_gen = ImageDataGenerator(validation_split=0.2,
                               rotation_range=30,
                               width_shift_range=0.10,
                               height_shift_range=0.10,
                               rescale=1/255,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest')

train_generator = image_gen.flow_from_directory(
    train_data_dir,
    shuffle=True,
    seed=42,
    target_size=image_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    subset='training', # set as training data
    classes=['Uninfected', 'Parasitized'])

validation_generator = image_gen.flow_from_directory(
    train_data_dir, # same directory as training data
    shuffle=False,
    seed=42,
    target_size=image_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    subset='validation', # set as validation data
    classes=['Uninfected', 'Parasitized'])


print(train_generator.class_indices)
print(validation_generator.class_indices)


# taking an image to check the transformation result

images = [i for i in os.listdir(folder_parasitized) if i.endswith(".png")]
img=imread('C:\\Users\\keert\\Datasets\\cell_images\\train\\Parasitized/' + images[0])
plt.imshow(img)


plt.imshow(image_gen.random_transform(img))



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



model.summary()

# set early stopping, to avoid overtraining

early_stop = EarlyStopping(monitor='val_loss',patience=3)




results = model.fit(train_generator,
                    epochs=10,
                    validation_data = validation_generator,
                    callbacks=[early_stop])




losses = pd.DataFrame(model.history.history)



losses[['loss','val_loss']].plot()



losses[['accuracy','val_accuracy']].plot()



model.evaluate(validation_generator)



pred_probabilities = model.predict(validation_generator)
pred_probabilities



predictions = pred_probabilities > 0.5
predictions



print(classification_report(validation_generator.classes, predictions))




confusion_matrix = pd.DataFrame(confusion_matrix(validation_generator.classes, predictions))
confusion_matrix



model.save('model111.h5')



# taking an sample image from dataset

image_shape = (130,130,3)
images = [i for i in os.listdir(folder_parasitized) if i.endswith(".png")]
img = 'C:\\Users\\keert\\Datasets\\cell_images\\train\\parasitized/' + images[2]




new_image = image.load_img(img, target_size=image_shape)
new_image


# prepare image

new_image = image.img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)


model.predict(new_image)
