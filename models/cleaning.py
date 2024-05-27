import numpy as np
import pandas as pd
from math import isnan

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

t_size = .01
rndom_seed = 40
cols = ["Sex", "Pclass", "Parch"]

# Data Reading

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Encoding Variables

train_df.Sex = train_df.Sex.apply(sex_encoding)
train_df.Embarked = train_df.Embarked.apply(embarked_encoding)
test_df.Sex = test_df.Sex.apply(sex_encoding)
test_df.Embarked = test_df.Embarked.apply(embarked_encoding)


# Handling Missing Values

if "Embarked" in cols:
    train_df = train_df.dropna(subset = ["Embarked"])
    test_df = test_df.dropna(subset = ["Embarked"])
if "Age" in cols:
    train_df = train_df.dropna(subset = ["Age"])
    test_df = test_df.dropna(subset = ["Age"])
if "Cabin" in cols:
    train_df.Cabin = train_df.Cabin.apply(cabin_encoding)
    test_df.Cabin = test_df.Cabin.apply(cabin_encoding)

# Name change, Check later

train = train_df
test = test_df
