from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
import soundfile
import librosa
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

cascPath = "C:\\Users\\Saivarma\\Desktop\\project\\StressDetection and EmotionDetection\\StressDetection\\model\\haarcascade_frontalface_default.xml"
#cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
face_emotion = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

main = tkinter.Tk()
main.title("Stress Detection from Images and Camera using Deep Learning Algorithm") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y, rf_model, tfidf_vectorizer
global face_classifier
global speech_X, speech_Y
global speech_classifier
global accuracy, precision, recall, fscore
global speech_X_train, speech_X_test, speech_y_train, speech_y_test
global image_X_train, image_X_test, image_y_train, image_y_test
stop_words = set(stopwords.words('english'))

def getID(name):
    index = 0
    for i in range(len(names)):
        if names[i] == name:
            index = i
            break
    return index        
    

def uploadDataset():
    global filename, tfidf_vectorizer
    filename = filedialog.askdirectory(initialdir=".")
    f = open('model/tfidf.pckl', 'rb')
    tfidf_vectorizer = pickle.load(f)
    f.close()  
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
    
def processDataset():
    text.delete('1.0', END)
    global X, Y, text_X, text_Y
    global speech_X, speech_Y
    global speech_X_train, speech_X_test, speech_y_train, speech_y_test
    global image_X_train, image_X_test, image_y_train, image_y_test
    global text_X_train, text_X_test, text_y_train, text_y_test
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        speech_X = np.load('model/speechX.txt.npy')
        speech_Y = np.load('model/speechY.txt.npy')
        text_X = np.load("model/textX.txt.npy")
        text_Y = np.load("model/textY.txt.npy")
        indices = np.arange(text_X.shape[0])
        np.random.shuffle(indices)
        text_X = text_X[indices]
        text_Y = text_Y[indices]
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32,32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3)
                    X.append(im2arr)
                    Y.append(getID(name))        
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    image_X_train, image_X_test, image_y_train, image_y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total number of Depression images found in dataset is  : "+str(len(X))+"\n")
    text.insert(END,"Dataset Train & Test Split\n\n")
    text.insert(END,"80% images used to train Deep Learning Algorithm : "+str(image_X_train.shape[0])+"\n")
    text.insert(END,"20% images used to test Deep Learning Algorithm : "+str(image_X_test.shape[0])+"\n")
    text_X_train, text_X_test1, text_y_train, text_y_test1 = train_test_split(text_X, text_Y, test_size=0.1)

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()


def trainFaceCNN():
    global face_classifier, accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    global image_X_train, image_X_test, image_y_train, image_y_test
    text.delete('1.0', END)
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            face_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        face_classifier.load_weights("model/cnnmodel_weights.h5")
        #face_classifier._make_predict_function()                  
    else:
        face_classifier = Sequential()
        face_classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
        face_classifier.add(MaxPooling2D(pool_size = (2, 2)))
        face_classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        face_classifier.add(MaxPooling2D(pool_size = (2, 2)))
        face_classifier.add(Flatten())
        face_classifier.add(Dense(output_dim = 256, activation = 'relu'))
        face_classifier.add(Dense(output_dim = 7, activation = 'softmax'))
        print(face_classifier.summary())
        face_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = face_classifier.fit(image_X_train, image_y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        face_classifier.save_weights('model/cnnmodel_weights.h5')            
        model_json = face_classifier.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/cnnhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = face_classifier.predict(image_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(image_y_test, axis=1)
    calculateMetrics("CNN Image Algorithm", predict, y_test1)   

def predictFaceDepression():
    global face_classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = face_classifier.predict(img)
    predict = np.argmax(preds)
    output = "Stressed"
    if predict == 3 or predict == 4:
        output = "Non Stressed"    
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Facial Output : '+output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    #cv2.putText(img, 'Percentage : '+str(0.7), (75, 50),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1)
    cv2.imshow('Facial Output : '+output, img)
    cv2.waitKey(0)

def runWebCam():
    global face_classifier
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        height, width, channels = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                sub_face = img[y:y+h, x:x+w]
            sub_face = cv2.resize(sub_face, (32,32))
            im2arr = np.array(sub_face)
            im2arr = im2arr.reshape(1,32,32,3)
            sub_face = np.asarray(im2arr)
            sub_face = sub_face.astype('float32')
            sub_face = sub_face/255
            preds = face_classifier.predict(sub_face)
            predict = np.argmax(preds)
            output = "Stressed"
            if predict == 3 or predict == 4:
                output = "Non Stressed" 
            cv2.putText(img, 'Facial Expression Recognized as : '+output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
            cv2.putText(img, 'Stress Feelings Recognized as : '+face_emotion[predict], (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.imshow('output', img)
            
        if cv2.waitKey(650) & 0xFF == ord('q'):
            break   
    cap.release()
    cv2.destroyAllWindows()
    
def graph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['Image CNN','Accuracy',accuracy[0]],['Image CNN','Precision',precision[0]],['Image CNN','Recall',recall[0]],['Image CNN','FSCORE',fscore[0]],
            ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()

def exit():
    main.destroy()

font = ('times', 13, 'bold')
title = Label(main, text='Stress Detection from Image using Deep Learning Algorithm')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=420,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Stress Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Train Facial Stress CNN Algorithm", command=trainFaceCNN)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 


graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

predictfaceButton = Button(main, text="Predict Facial Stress", command=predictFaceDepression)
predictfaceButton.place(x=50,y=300)
predictfaceButton.config(font=font1)

predictfaceButton = Button(main, text="Facial Stress from Cam", command=runWebCam)
predictfaceButton.place(x=50,y=350)
predictfaceButton.config(font=font1)

ExitButton = Button(main, text="Close GUI", command=exit)
ExitButton.place(x=50,y=400)
ExitButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
