import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
model = RandomForestClassifier()
print(train.head())
print(test.head())

y = train['label']
X = train.drop(labels=['label'], axis=1)
# drop the label

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# split the data

model.fit(X_train, y_train)

predict1 = model.predict(X_train)
predict2 = model.predict(X_val)

cm = metrics.confusion_matrix(y_train, predict1)
print(cm)

acc_train = accuracy_score(y_train, predict1)
acc_val = accuracy_score(y_val, predict2)

print(acc_train)
print(acc_val)

submission = pd.Series(predict2, name="Label")
# add the label column

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), submission], axis=1)
submission.to_csv("submission_prediction.csv", index=False)
