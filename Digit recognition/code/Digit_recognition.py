import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

training_data = pd.read_csv(r"Digit recognition\mnist\mnist_train.csv")
training_data = np.array(training_data)
training_data = training_data.T
labels = training_data[0]
images = training_data[1:training_data.size] / 255

testing_data = pd.read_csv(r"Digit recognition\mnist\mnist_test.csv")
testing_data = np.array(testing_data)
testing_data = testing_data.T
testing_labels = testing_data[0]
testing_images = testing_data[1:testing_data.size] / 255

def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    return np.exp(z - np.max(z, axis=0)) / np.sum(np.exp(z - np.max(z, axis=0)))

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(labels):
    labels = list(labels)
    one_hot_encoded = []
    for num in labels:
        vector = [0] * 10
        vector[num] = 1
        one_hot_encoded.append(vector)
    return np.array(one_hot_encoded).T

def back_prop(w2, a1, a2, images, one_hot_labels):
    m = images.size

    dz2 = a2 - one_hot_labels
    dw2 = dz2.dot(a1.T) / m
    db2 = np.sum(dz2) / m

    dz1 = w2.T.dot(dz2) * (a1 > 0)
    dw1 = dz1.dot(images.T) / m
    db1 = np.sum(dz1) / m

    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, axis=0)

def accuracy(predictions, labels):
    num_correct = np.sum(predictions == labels)
    return  num_correct / labels.shape[0]

# displays data about the progress of the neural network in the user interface
def data(iterations_list, accuracy_list, accuracy_graph_frame, accuracy_data_frame, a2, iteration, testing_a2, testing_accuracy_list):
    container = Figure(figsize=(4,3), dpi=100)
    graph = container.add_subplot(111)
    graph.plot(iterations_list, accuracy_list, label="Training data accuracy")
    graph.plot(iterations_list, testing_accuracy_list, label="Testing data accuracy")
    graph.set_ylabel("Accuracy")
    graph.set_xlabel("Iteration")
    graph.legend()
    container.tight_layout()
    canvas = FigureCanvasTkAgg(container, master=accuracy_graph_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

    tk.Label(accuracy_data_frame, text=f"Iteration {iteration}: Training data accuracy = {round(accuracy(get_predictions(a2), labels)*100, 1)}%", font=("Helvetica", 15)).grid(row=0, column=0, padx=10, pady=10)
    tk.Label(accuracy_data_frame, text=f"Iteration {iteration}: Testing data accuracy = {round(accuracy(get_predictions(testing_a2), testing_labels)*100, 1)}%", font=("Helvetica", 15)).grid(row=1, column=0, padx=10)

def gradient_descent(images, labels, iterations, alpha, accuracy_data_frame, accuracy_graph_frame):
    w1, b1, w2, b2 = init_params()

    iterations_list = []
    accuracy_list = []
    testing_accuracy_list = []

    for iteration in range(1, iterations+1):

        _, a1, _, a2 = forward_prop(w1, b1, w2, b2, images)
        _, _, _, testing_a2 = forward_prop(w1, b1, w2, b2, testing_images)

        dw1, db1, dw2, db2 = back_prop(w2, a1, a2, images, one_hot(labels))
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        iterations_list.append(iteration)
        accuracy_list.append(accuracy(get_predictions(a2), labels))
        testing_accuracy_list.append(accuracy(get_predictions(testing_a2), testing_labels))

        data(iterations_list, accuracy_list, accuracy_graph_frame, accuracy_data_frame, a2, iteration, testing_a2, testing_accuracy_list)
    
    return w1, b1, w2, b2
  
def make_prediction(index, w1, b1, w2, b2):
    image = testing_images[:, index, None]
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, image)
    prediction = get_predictions(a2)
    actual = testing_labels[index]
    confidence = a2.tolist()[prediction[0]]
    return prediction, actual, confidence

def make_prediction_on_drawn_image(img_array, w1, b1, w2, b2):
    _, _, _, a2 = forward_prop(w1, b1, w2, b2, img_array)
    prediction = get_predictions(a2)
    confidence = a2.tolist()[prediction[0]]
    return prediction, confidence