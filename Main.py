from tkinter import *
from tkinter import filedialog, simpledialog
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

main = Tk()
main.title("Wine Quality Prediction")
main.geometry("1300x1200")

global filename, dataset, X_train, X_test, y_train, y_test, scaler, rf_model, cnn_model
global accuracy, precision, recall, fscore

precision = []
recall = []
fscore = []
accuracy = []

labels = ["quality", "alcohol"]

def uploadDataset():
    text.delete('1.0', END)
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset.head()))
    dataset.hist(bins=20, figsize=(10, 10))
    plt.show()

def preprocessDataset():
    text.delete('1.0', END)
    global dataset, X, y, scaler, X_train, X_test, y_train, y_test
    dataset.fillna(0, inplace=True)
    X = dataset.drop(columns=["quality"])
    y = dataset["quality"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    text.insert(END, "Normalized Features\n")
    text.insert(END, str(X))
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    text.insert(END, "\n\nDataset Train & Test Split Details\n")
    text.insert(END, "80% dataset for training : " + str(X_train.shape[0]) + "\n")
    text.insert(END, "20% dataset for testing  : " + str(X_test.shape[0]) + "\n")

    plt.figure(figsize=(12, 12))
    sns.heatmap(dataset.corr() > 0.7, annot=True, cbar=False)
    plt.show()

def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict, average='macro', zero_division=1) * 100
    r = recall_score(testY, predict, average='macro', zero_division=1) * 100
    f = f1_score(testY, predict, average='macro', zero_division=1) * 100
    a = accuracy_score(testY, predict) * 100     
    text.insert(END, algorithm + ' Accuracy  : ' + str(a) + "\n")
    text.insert(END, algorithm + ' Precision   : ' + str(p) + "\n")
    text.insert(END, algorithm + ' Recall      : ' + str(r) + "\n")
    text.insert(END, algorithm + ' FMeasure    : ' + str(f) + "\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(testY, predict)
    
    # Plot confusion matrix
    plt.figure(figsize=(5, 5)) 
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis")
    plt.title(algorithm + " Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def trainLogisticRegression():
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test, accuracy, precision, recall, fscore, lr_model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    predict = lr_model.predict(X_test)
    calculateMetrics("Existing Logistic Regression", predict, y_test)
    
def trainKNN():
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test, accuracy, precision, recall, fscore
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test)
    calculateMetrics("Existing KNN", predict, y_test)

def trainDecisionTree():
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test, accuracy, precision, recall, fscore
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    calculateMetrics("Decision Tree", predict, y_test)

def trainRandomForest():
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test, accuracy, precision, recall, fscore, rf_model
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    predict = rf_model.predict(X_test)
    calculateMetrics("Extension Random Forest", predict, y_test)

def trainCNN():
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test, accuracy, precision, recall, fscore, cnn_model

    # Reshape the data to fit the CNN model (assuming input shape is (12, 1, 1))
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
    y_train_cnn = to_categorical(y_train)
    y_test_cnn = to_categorical(y_test)

    # Define an improved CNN model
    cnn_model = Sequential()
    cnn_model.add(Conv2D(64, (2, 1), activation='relu', padding='same', input_shape=(X_train.shape[1], 1, 1)))
    cnn_model.add(MaxPooling2D(pool_size=(2, 1)))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, (2, 1), activation='relu', padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 1)))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(y_train_cnn.shape[1], activation='softmax'))

    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Add checkpoints to save the best model
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Train the CNN model
    history = cnn_model.fit(X_train_cnn, y_train_cnn, epochs=100, batch_size=16, verbose=1, validation_data=(X_test_cnn, y_test_cnn), callbacks=callbacks_list)

    # Load the best model
    cnn_model.load_weights('best_model.h5')

    # Predict using the CNN model
    predict = cnn_model.predict(X_test_cnn)
    predict_classes = np.argmax(predict, axis=1)
    true_classes = np.argmax(y_test_cnn, axis=1)

    calculateMetrics("Proposed CNN", predict_classes, true_classes)

def comparisongraph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','F1 Score',fscore[1]],['KNN','Accuracy',accuracy[1]],
                       ['Decision Tree','Precision',precision[2]],['Decision Tree','Recall',recall[2]],['Decision Tree','F1 Score',fscore[2]],['Decision Tree','Accuracy',accuracy[2]],
                       ['Random Forest','Precision',precision[3]],['Random Forest','Recall',recall[3]],['Random Forest','F1 Score',fscore[3]],['Random Forest','Accuracy',accuracy[3]],
                       ['CNN','Precision',precision[4]],['CNN','Recall',recall[4]],['CNN','F1 Score',fscore[4]],['CNN','Accuracy',accuracy[4]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()

def predictWithRandomForest():
    text.delete('1.0', END)
    global scaler, rf_model
    if rf_model is None:
        text.insert(END, "Please train the Random Forest model first.\n")
        return

    # Get user input
    input_values = []
    input_features = dataset.drop(columns=["quality"]).columns
    for feature in input_features:
        value = simpledialog.askfloat("Input", f"Enter value for {feature}:")
        input_values.append(value)

    # Preprocess input
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Predict using Random Forest
    prediction = rf_model.predict(input_scaled)
    text.insert(END, f"Predicted quality: {prediction[0]}\n")

# UI Elements
font = ('times', 16, 'bold')
title = Label(main, text='Prediction Of Wine-Quality Using Multi Level Trained Model')
title.config(bg='White', fg='Maroon1')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=27, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=200)
text.config(font=font1)

uploadButton = Button(main, text="Upload Wine Dataset", command=uploadDataset)
uploadButton.place(x=10, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=250, y=100)
preprocessButton.config(font=font1)

svmButton = Button(main, text="Logistic Regression Algorithm", command=trainLogisticRegression)
svmButton.place(x=490, y=100)
svmButton.config(font=font1)

knnButton = Button(main, text="Train KNN Algorithm", command=trainKNN)
knnButton.place(x=730, y=100)
knnButton.config(font=font1)

dtButton = Button(main, text="Train Decision Tree Algorithm", command=trainDecisionTree)
dtButton.place(x=970, y=100)
dtButton.config(font=font1)

rfButton = Button(main, text="Train Random Forest Algorithm", command=trainRandomForest)
rfButton.place(x=10, y=150)
rfButton.config(font=font1)

cnnButton = Button(main, text="Train CNN Algorithm", command=trainCNN)
cnnButton.place(x=250, y=150)
cnnButton.config(font=font1)


graphButton = Button(main, text="Comparision Graph", command=comparisongraph)
graphButton.place(x=490, y=150)
graphButton.config(font=font1)


predictButton = Button(main, text="Predict The Wine Quality", command=predictWithRandomForest)
predictButton.place(x=730, y=150)
predictButton.config(font=font1)

main.config(bg='Maroon1')
main.mainloop()
