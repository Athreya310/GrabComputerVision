import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

def load_image(train_path, valid_path):
    # Change directory to the data folder
    os.chdir(x + r'\compvision\Raw Data')
    # This is image augmentation. It alters the images creating a larger sample to test on. This reduces overfitting
    # Rescale pixel values between 0 and 1 instead of the regular 255
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    valid_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size = (128, 128),
                                                     batch_size = 32,
                                                     class_mode = 'categorical')
    
    valid_set = valid_datagen.flow_from_directory(valid_path,
                                                target_size = (128, 128),
                                                batch_size = 32,
                                                class_mode = 'categorical')
    return(training_set,valid_set)


def create_classifier(activ, drop, opt, ls, mets):
    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Convolution2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(output_dim = 512, activation = 'relu'))
    classifier.add(Dense(output_dim = 196, activation = activ))
    
    # Step 5 - Dropout
    classifier.add(Dropout(drop))
    
    # Compiling the CNN
    classifier.compile(optimizer = opt, loss = ls, metrics = mets)
    
    return(classifier)
    
    
def map_classifier(classifier, train, validation):
    # Fitting the model to the training set
    classifier.fit_generator(train,
                    steps_per_epoch = 200,
                    epochs = 15,
                    validation_data = validation,
                    validation_steps = 50)

def save_classifier(classifier):
    # serialize model to JSON
    # Change directory to the model specs folder
    os.chdir(x + r'\compvision\Model Specs')
    model_json = classifier.to_json()
    with open("modelvar.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("modelvar.h5")
    print("Saved model to disk")    


x = r'C:\Users\ASUS\Desktop\Data Science\Grab challenge'

if __name__ == '__main__':
    
    tr,val = load_image('data/training_set','data/validation_set')
    c = create_classifier(activ = 'softmax', drop = 0.2, opt = 'adam', ls = 'categorical_crossentropy', mets = ['accuracy'])
    map_classifier(c,tr,val)
    save_classifier(c)