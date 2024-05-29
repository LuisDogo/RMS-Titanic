import numpy as np
import pandas as pd
from math import isnan
from sklearn.model_selection import train_test_split

def sex_encoding(sex):
    if sex == "male":
        return 1
    else:
        return 0

def embarked_encoding(port): # By Order
    if port == "S":
        return 1
    elif port == "C":
        return 2
    elif port == "Q":
        return 3

def cabin_encoding(cabin):
    if type(cabin) == str:
        return 0
    else:
        return 1

# Universal Variables for the models to use

t_size = .20
rndom_seed = 40
cols = ["Sex", "Pclass", "Parch"]

# Data Reading

train = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Encoding Variables

train.Sex = train.Sex.apply(sex_encoding)
train.Embarked = train.Embarked.apply(embarked_encoding)
test_df.Sex = test_df.Sex.apply(sex_encoding)
test_df.Embarked = test_df.Embarked.apply(embarked_encoding)


# Handling Missing Values

if "Embarked" in cols:
    train = train.dropna(subset = ["Embarked"])
    test_df = test_df.dropna(subset = ["Embarked"])
if "Age" in cols:
    train = train.dropna(subset = ["Age"])
    test_df = test_df.dropna(subset = ["Age"])
if "Cabin" in cols:
    train.Cabin = train.Cabin.apply(cabin_encoding)
    test_df.Cabin = test_df.Cabin.apply(cabin_encoding)

class_names = ["Survived", "Died"]

# Split data

X_train, X_test, y_train, y_test = train_test_split(train[cols], train.Survived, test_size = t_size, random_state = rndom_seed)

