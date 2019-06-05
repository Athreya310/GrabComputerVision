import scipy.io
import numpy as np
import os
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing import image
import time

def generate_car_names():
    # Change directory to the data folder
    os.chdir(x + r'\compvision\Raw Data')
    
    # Create a list with the model names
    list_model = []
    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)
    for i in range(196):
        model = class_names[i][0][0]
        list_model.append(model)
    return(list_model)

def load_classifier():
    # Change directory to the model specs folder
    os.chdir(x + r'\compvision\Model Specs')
    # Load json and create model
    json_file = open('modelvar.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # Loading the model
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("modelvar.h5")
    print("Loaded model from disk")
    return(classifier)

def output_classifier(list_model,classifier,top_n):
    df = pd.DataFrame(columns=['Car Make','Probability','Alt1','Alt2'])
    # Change directory to the test images folder
    os.chdir(x + r'\compvision\Raw Data\cars_test')
    filelist=os.listdir()
    # For each image in the test folder
    for j in filelist:
        test_image = image.load_img(j, target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        test_image /= 255
        result_prob = classifier.predict(test_image)
        # k returns the highest probability
        k = np.amax(result_prob)
        k = round(k*100,1)
        # Returns top n car mak predictions
        top = result_prob[0].argsort()[::-1][:top_n]
        toptop = []
        # Places the top n predictions in their full model name form in toptop list
        for t in top:
            t = list_model[t]
            t = str(t)
            toptop.append(t)
        df = df.append({'Car Make':toptop[0],'Probability':k, 'Alt1':toptop[1], 'Alt2':toptop[2]}, ignore_index = True)
    return(df)
    
# Set x as the working directory to the folder where the files are saved.
x = r'C:\Users\ASUS\Desktop\Data Science\Grab challenge'

if __name__ == '__main__':
    list_model = generate_car_names()
    c = load_classifier()
    df3 = output_classifier(list_model,c,3)


# User interface
while True:
    try:
        re = input('Which image would you like to test?\nExample: 1,2,3....8041\nType \'stop\' to end program.\n')
        # Break out of loop when done
        if re == 'stop':
            break
        # Response in high certainty cases
        elif df3['Probability'][int(re)-1] > 70:
            # Turn it to its appropriate index form
            re = int(re)-1
            print('I am quite certain. The car in the image is quite likely the',df3['Car Make'][re], '(',df3['Probability'][re],'% chance). Alternatively, it could be the',
            df3['Alt1'][re], 'or the', df3['Alt2'][re],'.')
            time.sleep(5)
        # Response in low certainty case
        elif df3['Probability'][int(re)-1] <30:
            # Turn it to its appropriate index form
            re = int(re)-1
            print('I am highly uncertain. My best guess is that the car in the image is most likely the',df3['Car Make'][re], '(',df3['Probability'][re],'% chance). Alternatively, it could be the',
            df3['Alt1'][re], 'or the', df3['Alt2'][re],'.')
            time.sleep(5)
        # Response in mid-certainty cases
        else:
            # Turn it to its appropriate index form
            re = int(re)-1
            print('I am slightly uncertain. I think the car in the image is most likely the',df3['Car Make'][re], '(',df3['Probability'][re],'% chance). Alternatively, it could be the',
                  df3['Alt1'][re], 'or the', df3['Alt2'][re],'.')
            time.sleep(5)
                   
    except:
        print('Please try again.')
        time.sleep(3)