import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cleaning import test, train, t_size, rndom_seed, cols

X_train, X_test, y_train, y_test = train_test_split(train[cols], train.Survived, test_size = t_size, random_state = rndom_seed)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)