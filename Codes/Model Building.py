import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

# USER INPUT REQUIRED - Set x as the working directory to the folder where the folder downloaded from github is saved.
x = r'C:\Users\ASUS\Documents'

# This function loads images from the folder and prepares it for the CNN. 
# Function arguments: train_path - path to training set class folders, valid_path - path to validation set class folders
def load_image(train_path, valid_path):
    print('Loading images')
    # Change directory to the data folder
    os.chdir(x + r'\GrabComputerVision\Raw Data')
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

# Creates the CNN classifier
# Function arguments: activ - activation function for final dense layer, drop - dropout value, opt - optimizer algorithm, ls - loss function, mets - performance metrics 
def create_classifier(activ, drop, opt, ls, mets):
    print('Creating classifier')
    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dense(units = 196, activation = activ))
    
    # Step 5 - Dropout
    classifier.add(Dropout(drop))
    
    # Compiling the CNN
    classifier.compile(optimizer = opt, loss = ls, metrics = mets)
    
    return(classifier)
    
# Maps the CNN to the training set
# Function arguments: classifier - classifier model name, train - processed training set from first function, valid - processed validation set from first function
def map_classifier(classifier, train, validation):
    print('Mapping classifier')
    # Fitting the model to the training set
    classifier.fit_generator(train,
                    steps_per_epoch = 200,
                    epochs = 15,
                    validation_data = validation,
                    validation_steps = 50)
    
# Saves the CNN model and weights
# Function arguments: classifier - classifier model name
def save_classifier(classifier):
    # serialize model to JSON
    # Change directory to the model specs folder
    os.chdir(x + r'\GrabComputerVision\Model Specs')
    model_json = classifier.to_json()
    with open("modelvar.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("modelvar.h5")
    print("Saved model to disk")    

if __name__ == '__main__':
    
    tr,val = load_image('data/training_set','data/validation_set')
    c = create_classifier(activ = 'softmax', drop = 0.2, opt = 'adam', ls = 'categorical_crossentropy', mets = ['accuracy'])
    map_classifier(c,tr,val)
    save_classifier(c)