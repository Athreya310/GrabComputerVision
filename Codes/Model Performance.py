import numpy as np
import os
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# USER INPUT REQUIRED - Set x as the working directory to the folder where the folder downloaded from github is saved.
x = r'C:\Users\ASUS\Documents'

def load_classifier():
    # Change directory to the model specs folder
    os.chdir(x + r'\GrabComputerVision\Model Specs')
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
 
# This is the most basic classifier. Output includes the confusion matrix, scores of accuracy, precision and recall. To study model performance only.
def basic_classifier(classifier):
    print('Executing basic classifier...')
    # Create dataframe to store predicted car model and the true car model
    df = pd.DataFrame(columns=['Predicted','True'])
    # Change directory to the data folder
    os.chdir(x + r'\GrabComputerVision\Raw Data')
    for i in range(196):
        # This rectifies index to be identical to class label.
        i = i + 1
        i = "%04d" % i
        # Retrieve all images in the class file
        filelist=os.listdir(r'data\validation_set\{}'.format(i))
        # For each image in the class folder
        for j in filelist:
            string = x + r'\GrabComputerVision\Raw Data\data\validation_set\{}\{}'.format(i,j)
            # Load and process images to be tested
            test_image = image.load_img(string, target_size=(128,128))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image,axis=0)
            test_image /= 255
            # Apply model to images to be tested
            # Returns the class predicted by the model
            result_class = classifier.predict_classes(test_image)
            result_class = result_class[0] + 1
            result_class = "%04d" % result_class
            # Returns a dataframe of predicted and true classes of cars for each image.
            df = df.append({'Predicted':result_class,'True':i}, ignore_index = True)
    # Create an array of prediction and true values to generate the performance scores
    pred = df.as_matrix(columns = ['Predicted'])
    true = df.as_matrix(columns = ['True'])
    # Standard performance measures
    matrix = confusion_matrix(pred,true)
    accuracy = accuracy_score(pred,true)
    precision = precision_score(pred,true,average='macro')
    recall = recall_score(pred,true,average='macro')

    return(df,matrix,accuracy,precision,recall)


# The threshold sets the lower bound of confidence in images shown. 
# E.g. 0.8 shows accuracy of results where the model was at least 80% certain of prediction.
def threshold_classifier(threshold,classifier):  
    print('Executing threshold classifier...')
    # Create dataframe to store results
    df = pd.DataFrame(columns=['Car Make','Accuracy'])
    # Change directory to the data folder
    os.chdir(x + r'\GrabComputerVision\Raw Data')
    # This is the total images of all classes
    total_count = 0
    # This is the total images predicted accurately
    total_positive = 0
    
    for i in range(196):
        # Total number of images in the class.
        count = 0
        # Total number of images predicted accurately in the class.
        positive = 0
        # This rectifies index to be identical to class label.
        i = i + 1
        i = "%04d" % i
        # Retrieve all images in the class file
        filelist=os.listdir(r'data\validation_set\{}'.format(i))
        # For each image in the class folder
        for j in filelist:
            string = x + r'\GrabComputerVision\Raw Data\data\validation_set\{}\{}'.format(i,j)
            # Load and process images to be tested
            test_image = image.load_img(string, target_size=(128,128))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image,axis=0)
            test_image /= 255
            # Apply model to images to be tested
            # Return probabilities of all classes
            result_prob = classifier.predict(test_image)
            # Returns the class predicted by the model
            result_class = classifier.predict_classes(test_image)
            result_class = result_class[0] + 1
            # set check value to the true model class integer
            check = int(i)
            # Return highest probability value
            k = np.amax(result_prob)
            # Only consider the accuracy when the probability passess the confidence threshold outlined. If it doesn't we ignore it from the positive and count counters.
            if k > threshold:
                if result_class == check:
                    positive += 1
                count +=1  
        # Accuracy of each class
        try:
            accuracy = positive / count
        except:
            accuracy = None
        df = df.append({'Car Make':i,'Accuracy':accuracy}, ignore_index = True)
        # Add counts and positives to the total count
        total_count = total_count + count
        total_positive = total_positive + positive
    # Calculates the overall accuracy of the model
    total_accuracy = total_positive / total_count
    return(threshold,df,total_accuracy)


# Produces results from the top n most likely car models.
# E.g. if top_n = 5, it produces the accuracy of one of the top 5 likeliest car models being the true model in the image
def top_classifier(top_n,classifier):   
    print('Executing top_n classifier...')
    # Create dataframe to store results
    dft = pd.DataFrame(columns=['Car Make','Accuracy'])
    # This is the total images of all classes
    total_count_top = 0
    # Total number of images predicted accurately in the class.
    total_positive_top = 0
    
    for i in range(196):
        # Total number of images in the class.
        count_top = 0
        # Total number of images predicted accurately in the class.
        positive_top = 0
        # This rectifies index to be identical to class label.
        i = i + 1
        i = "%04d" % i
        # Retrieve all images in the class file
        filelist=os.listdir(r'data\validation_set\{}'.format(i))
        for j in filelist:
            # Load and process images to be tested
            string = x + r'\GrabComputerVision\Raw Data\data\validation_set\{}\{}'.format(i,j)
            test_image = image.load_img(string, target_size=(128,128))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image,axis=0)
            test_image /= 255
            # Apply model to images to be tested
            result_prob = classifier.predict(test_image)
            # Returns the top 5 classes in order of their probability in index form.
            top5 = result_prob[0].argsort()[::-1][:top_n]
            # Test if one of the 5 classes predicted is the true class
            for t in top5:
                # Add 1 to make the index comparabe to class labels
                t += 1
                if t == int(i):
                    positive_top += 1    
            count_top += 1
        # Accuracy of each class
        try:
            accuracy_top = positive_top / count_top
        except:
            accuracy_top = None
        dft = dft.append({'Car Make':i,'Accuracy':accuracy_top}, ignore_index = True)
        # Add counts and positives to the total count
        total_count_top = total_count_top + count_top
        total_positive_top = total_positive_top + positive_top
    # Calculates the overall accuracy of the mdoel
    total_accuracy_top = total_positive_top / total_count_top
    
    return(top_n, dft,total_accuracy_top)
    
if __name__ == '__main__':
    c = load_classifier()
    df0,confusion_mat,score_accuracy,score_precision,score_recall = basic_classifier(c)
    threshold, df1, acc_thresh = threshold_classifier(0.8,c)
    top_models, df2, acc_top = top_classifier(5,c)

# Write results to a text file
print('Writing results to scores.txt')
os.chdir(x + r'\GrabComputerVision\Model Specs')
f= open("scores.txt","w+")
f.write('1. This is the basic performance evaluation of the validation set:\n')
f.write('-  The accuracy score is '+str(score_accuracy)+'.\n')
f.write('-  The precision score is '+str(score_precision)+'.\n')
f.write('-  The recall score is '+str(score_recall)+'.\n\n')
f.write('2. The threshold model can achieve an accuracy of '+str(acc_thresh)+' when the threshold is set to '+str(threshold)+'.\n\n')
f.write('3. The top_n model can achieve an accuracy of '+str(acc_top)+' when number of top models is set to '+str(top_models)+'.\n\n')
f.close()

