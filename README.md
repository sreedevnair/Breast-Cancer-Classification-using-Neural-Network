# Breast Cancer Classification using Neural Network
In this project, we will be creating a neural network that can predict whether the tumor is Benign (Non Cancerous) or Malignant (Cancerous).

We will be using scikit-learn for data processing and TensorFlow to create the neural network.

## Code by code explaination :-

### 1. Importing all the modules
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
```

Here, we are just importing all the neccessary modules for the data handling and pre processing.

<br>

### 2. Loading the dataset
```
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

print("Breast Cancer Data :-")
print(breast_cancer_dataset.data)

print("\nBreast Cancer Target Value :-")
print(breast_cancer_dataset.target)

print("\nBreast Cancer Target Name :-")
print(breast_cancer_dataset.target_names)

print("\nBreast Cancer Columns :-")
breast_cancer_dataset.feature_names
```
In the first line, we are loading the breast cancer dataset from `sklearn.datasets` into the variable *breast_cancer_dataset*.

Now the loaded dataset is in the form of a dictionary which contains the following **keys** along with their respective values :-
1. data
2. target
3. frame
4. target_names
5. DESCR
6. feature_names
7. filename
8. data_module

#### 2.1. "data" key
This key contains the actual **feature data** in the form of lists. Each list represents a single row or the features of a single tumor. 
This key-value pair doesn't contain the finale target value, i.e. whether the tumor is malignant or benign.

### 2.2 "target" key
The value of this key is a single list that contains the target value for each list from the *data* key. It is represented in the form of 0s and 1s, where 0 represents malignant and 1 represents benign.

### 2.3 "target_names" key
It correspond to categorical values of the target, i.e. `['malignant' 'benign']` in our case.
We will not use this inside our neural network.

### 2.4 "feature_names" key
It's an array of all the feature names.

<br>

## 3. Converting the dataset into a DataFrame
```
df = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
```
We are using `pd.DataFrame()` function to convert our loaded dataset in a pandas' DataFrame. It takes 2 arguments :-
1. The dataset (rows)
2. The column name (i.e. the column header)

This DataFrame still doesn't contain the target column yet.

<br>

## 4. Adding target column to our DataFrame
```
df['label'] = breast_cancer_dataset.target
```
We are creating a new column names **label** that will contain the target value of each row.

<br>

### 5. Analyzing our data
```
df.shape

df.isnull().sum()

df.describe()

df.label.value_counts()

df.groupby('label').mean()
```

#### 5.1. `df.shape`
This attribute returns a tuple with the size of the DataFrame, i.e. the number of rows and the number of columns

Our DataFrame contiains 569 rows and 31 columns (30 features and 1 target column)

#### 5.2 `df.isnull().sum()`
`.isnull()` returns the complete DataFrame with True or False values instead of their original values. If a value is not present (i.e. a Null value), it will show `True`, otherwise `False`.

To check the total number of Null values present in the DataFrame, we use `.isnull().sum()`. It returns a Series with the number of null values present in each column (feature).

#### 5.3 `df.describe()`
This functions gives us a quick overview of our data. It will tell us the total *count*, *mean*, *std* (standard deviation), *min* and *max*, *25 %ile*, *50 %ile* and *75 %ile* of each column of the dataframe

#### 5.4. `df.label.value_counts()`
We can also write this as `df['label'].value_counts()`. This function returns the number of occurences of each unique values present in that column.

#### 5.5. `df.groupby('label').mean()`
It **groups** our DataFrame by the unique values in the **label** column (e.g., *0* and *1*) and then it will calculate the mean of all other numerical columns for each group. 

It returns a new DataFrame showing the average values of those columns for the rows where 'label' is 0 and where 'label' is 1.

For example, one of the features in our dataset is *radius*. This function will calculate the average value of all the *radius* where the label is *0* and then it will calculate the average value of all the *radius* where the label is *1*.

<br>

### 6. Splitting our dataset
```
X = df.drop(columns='label')
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
```

First, we need to split our dataset into the feature set and the target set.

Then, we split our dataset further into training data and testing data using sklearn's `train_test_split` function.

We need to pass the feature set and the target set as the parameter. We can also pass other parameters :-

1. `test_size` : By default it's .25, which means 25% of the dataset will be used for testing and the other 75% for the training.
2. `random_state` : This is to make sure that our data split remains same during every execution.

<br>

### 7. Standardizing our data
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)
```

#### 7.1 `from sklearn.preprocessing import StandardScaler`
First, we are importing `StandardScaler` from `sklearn.preprocessing`. The `StandardScaler` is used to standardize features by removing the mean and scaling them to unit variance, which helps in improving the performance of many machine learning algorithms.

In simple terms, it does the following 2 things :-
1. **Removing the mean**: Each feature in the dataset (each column) is adjusted (subtracting the mean from each value) so that its mean becomes 0.
2. **Dividing by the standard deviation of each feature**: This makes the variance of each feature equal to 1, which helps ensure that all features contribute equally to the model.

So, the formula for standardization is : $$z$$ = $$\frac{x-\mu}{\sigma}$$


Where :-
- \(z\) is the standardized value.
- \(x\) is the original value.
- ($${\mu}$$) is the mean of the data.
- ($${\sigma}$$) is the standard deviation of the data.

#### 7.2 `scaler = StandardScaler()`
Here, we are creating an instance of the `StandardScaler` class.

#### 7.3 `X_train_std = scaler.fit_transform(X_train)`
The `fit_transform()` function does two things :-
1. **Fit**: It calculates the mean and standard deviation of the `X_train` data.
2. **Transform**: Then, it applies the standardization to the `X_train` data. The standardized training data is then stored in `X_train_std`.

Similarly, we also standardize the `X_test` data and stores the new data in `X_test_std`.

<br>

### 8. Building the Neural Network

#### 8.1. Importing TensorFlow
```
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
```
As usual, we are import TensorFlow to create the neural network.

The `tf.random.set_seed(3)` functions sets the global random seed for TensorFlow's random number generators to 3. By setting a seed, TensorFlow ensures that the random processes (like initializing weights in a neural network) will produce the same results every time we run the code. It's like using `random_state` when splitting our dataset.

#### 8.2. Creating the NN model
```
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(2, activation="sigmoid")
])
```

##### a. `keras.Sequential()`
This initializes a Sequential model which is a linear stack of layers. We can add layers to this model one by one, and the data will flow through these layers in the order they are added. This model is good for building a feedforward neural network, where the output of one layer is the input to the next.

##### b. `keras.layers.Flatten(input_shape=(30,))`
This layer doesn't contain neurons with weights and biases. Instead, it reshapes the input data. The **Flatten** layer converts the input data into a 1D array (flattening it). Here, the input shape is specified as (30,), meaning the model expects 30 features as input.

##### c. `keras.layers.Dense(20, activation="relu")`
A **dense** layer is a fully connected layer where each neuron receives input from all the neurons in the previous layer. It's the core layer of most neural networks. This adds a dense layer with 128 neurons to the model. 
The activation function applied to this layer is *ReLU* (Rectified Linear Unit). ReLU introduces non-linearity into the model, which helps the network learn complex patterns. It converts all negative values to zero and keeps positive values unchanged.

##### d. `keras.layers.Dense(2, activation="sigmoid")`
This adds another dense layer with only 2 neurons, because the model is predicting 2 possible classes (malignant and benign).
<br>
<br>
`activation="sigmoid"` outputs a probability between 0 and 1 for each of the two neurons. And which neuron has the highest probability is activated (outputed).

#### 8.3. Configuring the model for training
```
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
```
We use `model.compile()` function to configure the model like specifying the optimizer, the loss function, and the metrics that the model should track during training and evaluation.

##### a. `optimizer='adam'`
The optimizer is an algorithm that adjusts the weights and biases of the model in order to minimize the loss function during training. Here, we are using `adam` optimzer which adjust the learning rate adaptively for each parameter in the model based on the history of gradients calculated for that parameter :)

##### b. `loss='sparse_categorical_crossentropy'`
Loss function measures the difference between the predicted output and the actual target value. Our goal is to minimize loss function during the model training.

1. **sparse** : It indicates that the target labels are provided as integers rather than one-hot encoded vectors .
2. **categorical_crossentropy** : It measures how well the predicted probabilities (output of the sigmoid layer) match the actual labels.

So basically, sparse categorical crossentropy is used to measure how far the predictions are from the actual labels, where the labels are provided as integers.

##### c. `metrics=['accuracy']`
Metrics are used to evaluate the performance of the model during training and testing. **accuracy** is a common metric for classification problems. It calculates the percentage of correct predictions out of the total predictions. 
When *loss* decreases, the *metrics* value will increase.

#### 8.3. Training the model
```
history = model.fit(X_train_std, y_train, validation_split=0.1, epochs=10)
```

Here, we train the model using the standardized training set `(X_train_std, y_train)`.

##### a. **Epochs**
An epoch is one complete pass through the entire training dataset.<br>
`epochs=10`: This specifies that the model should go through the entire X_train_std and y_train datasets 10 times, adjusting its parameters after each pass.

##### b. **Validation Split**
During the training process, when you use `validation_split=0.1` in` model.fit()`, 10% of our training data ( X_train_std  and    y_train ) is reserved as the validation set. This validation set is used to evaluate the model’s performance after each epoch, but it’s not used to adjust the model’s parameters (weights and biases) like the training set is.

##### c. **history**
It stores information about the training process, such as loss and accuracy, which can be used to analyze how the model performed over time.

##### What Happens During `model.fit()`?
1. **Forward Pass on Training Data:**<br>
    a. The model makes predictions on the training data `X_train_std`.<br>
    b. Loss is calculated based on how far the predictions are from the true labels `y_train`.<br>
2. **Backward Pass (Gradient Descent):**<br>
    The optimizer (like Adam) adjusts the model's weights to minimize the loss based on training data.<br>
3. **Forward Pass on Validation Data:**<br>
    a. After each epoch, the model makes predictions on the *validation set*.<br>
    b. The validation loss and accuracy is calculated to check how well the model is performing on unseen data.<br>
    c. This helps detect overfitting and underfitting.<br>
4. **No Backward Pass on Validation Set:**<br>
    The model does not adjust its parameters based on the validation set. It’s purely used to monitor performance.<br>
5. **Repeat for Each Epoch:**<br>
    Training (with updates) happens on the training set, and after each epoch, the model is evaluated on the validation set.

##### How does the validation loss and accuracy helps in detecting overfitting ?
1. At some point, if the model gets too focused on the training data, it starts memorizing specific details or noise in the training images that don't generalize well.
2. The **training loss keeps decreasing**, but the **validation loss starts increasing**. This means the model is doing great on the training data but poorly on the validation data.
3. Similarly, the **training accuracy becomes very high**, but the **validation accuracy stops improving or gets worse**. This is overfitting.

### 9. Visualizing Accuracy and Loss
```
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

plt.legend(['Training', 'Validation'], loc = 'lower right')
```

```
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.legend(['Training', 'Validation'], loc = 'lower right')
```
Here, we are just plotting the **training accuracy and loss** and **validation accuracy and loss**.

### 10. Evaluating the model
```
loss, accuracy = model.evaluate(X_test_std, y_test)

print("Accuracy of the model is :", accuracy)
print("Loss of the model is :", loss)
```
Once the model is trained, we can evalute the model using the test dataset ```(X_test_std, y_test)```. This will return the model's prediction loss and its accuracy.

### 11. Predicting target values for the `X_test_std`
```
y_pred = model.predict(X_test_std)
```

```
print(y_pred.shape)
print(y_pred[0])
```
```
Output :-
>>> (114, 2)
>>> [0.22896117 0.5600809 ]
```
Here, instead of calculating the overall performance of the model on the `X_test_std` like we did above, we predict every **target value** for each sample. 

Since the testing set contains 142 rows, it will predict 142 target values. Each target value is a list made up of 2 elements, i.e. the probabilities of the two output neuron/classes (Malignant and Benign respectivly).

The neuron with the highest probability is activated.

```
print(y_pred)
```

```
Output :-
>>> [[2.28961170e-01 5.60080886e-01]
    [9.53364015e-01 5.67435734e-02]
    [9.12674606e-01 1.64341614e-01]
    ...............................
    [9.99702573e-01 3.24228755e-03]]
```

### Converting the prediction probability into class labels
```
y_pred_labels = [np.argmax(i) for i in y_pred ]

print(y_pred_labels)
```

```
Output :-
>>> [1, 0, 0, 1, 1, 0, 0, 0, 1, ........, 1, 1, 0, 1, 1, 0]
```

So what `np.argmax(i)` does is that, first we need to pass a list `i` as the parameter, and then it return the index number of the largest element present in that list.

Hence, if the first neuron (i.e. Malignant) has the highest probability, it will output **0** (which also represents Malignant in our *label* column). And if the second neuron (i.e. Benign) has the highest probability, it will output **1** (which represents Benign in our *label* column)


----

