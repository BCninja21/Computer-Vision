import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras.layers import Dense

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# Read the dataset

X_train = train.drop('label', axis=1)
labels = train.label
print(X_train.shape)
X_train = np.array(X_train) / 255
X_test = np.array(test) / 255
# Normalization

# Build the model
ann = Sequential()
ann.add(Dense(units=784, activation='relu'))
ann.add(Dense(units=128, activation='relu'))
ann.add(Dense(units=64, activation='relu'))
ann.add(Dense(units=32, activation='relu'))
ann.add(Dense(units=10, activation='softmax'))
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = ann.fit(x=X_train, y=labels, batch_size=32, epochs=10, validation_split=0.2)

val_loss = hist.history['val_loss']
val_accuracy = hist.history['val_accuracy']
train_loss = hist.history['loss']
train_accuracy = hist.history['accuracy']

# Visualization
sns.set_style('darkgrid')
sns.lineplot(train_accuracy, label='train_accuracy')
sns.lineplot(val_accuracy, label='val_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.title('Accuracy train & validation')
plt.show()

sns.set_style('darkgrid')
sns.lineplot(val_loss, label='val_loss')
sns.lineplot(train_loss, label='train_loss')
plt.ylabel('loss')
plt.xlabel('Number of epochs')
plt.title('loss train & validation')
plt.show()

# Output the prediction
predictions = ann.predict(test)
predicted_labels = np.argmax(predictions, axis=1)
digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
predicted_digit_labels = [digit_labels[i] for i in predicted_labels]
image_id = list(range(1, len(predicted_digit_labels) + 1))
submission_df = pd.DataFrame({'ImageId': image_id, 'Label': predicted_digit_labels})
submission_df.to_csv('submission.csv',index=False)