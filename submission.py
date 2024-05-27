import csv
import numpy as np
from models.cleaning import test, train, t_size, rndom_seed, cols
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train[cols], train.Survived, test_size = t_size, random_state = rndom_seed)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(test[cols])
rows = []
for i in range(0, len(y_pred)):
    rows.append([test.PassengerId[i], y_pred[i]])

print(rows)

# field names
fields = ['PassengerId', 'Survived']
 
# name of csv file
filename = "submission.csv"
 
# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
 
    # writing the fields
    csvwriter.writerow(fields)
 
    # writing the data rows
    csvwriter.writerows(rows)