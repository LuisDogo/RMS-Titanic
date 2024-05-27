import numpy as np
import pandas as pd
from scipy.stats import entropy

def space():
    print("\n\n")

def prob_df(df):
    a = df.iloc[0] / df.iloc[2]
    b = df.iloc[1] / df.iloc[2]
    return a, b

def per(f): # convierte a porcentaje en este fomrmato XX.XXX%
    return str(('%.3f'%(f * 100))) + "%"

def prob(a, b, c):
    return [a/c, b/c]

def conditional_probs(df):
    survivors = df[df.Survived == 1]
    victims = df[df.Survived == 0]
    male = df[df.Sex == "male"]
    female = df[df.Sex == "female"]
    class_1 = df[df.Pclass == 1]
    class_2 = df[df.Pclass == 2]
    class_3 = df[df.Pclass == 3]
    has_cabin = df[~pd.isna(df.Cabin)]
    u = len(df)
    class_1_survivors = pd.merge(survivors, class_1, how = "inner")
    class_2_survivors = pd.merge(survivors, class_2, how = "inner")
    class_3_survivors = pd.merge(survivors, class_3, how = "inner")
    male_survivors = pd.merge(survivors, male, how = "inner")
    female_survivors = pd.merge(survivors, female, how = "inner")
    print(len(male_survivors) / len(male))
    print(len(female_survivors) / len(female))
    print(len(class_1_survivors) / len(class_1))
    print(len(class_2_survivors) / len(class_2))
    print(len(class_3_survivors) / len(class_3))
    f_class_1_survivors = pd.merge(class_1_survivors, female, how = "inner")
    f_class_1 = pd.merge(class_1, female, how = "inner")
    print(len(f_class_1_survivors) / len(f_class_1))

    m_class_3_survivors = pd.merge(class_3_survivors, male, how = "inner")
    m_class_3 = pd.merge(class_3, male, how = "inner")
    print(len(m_class_3_survivors) / len(m_class_3))

    has_cabin_survivors = pd.merge(has_cabin, survivors, how = "inner")
    print(len(has_cabin_survivors) / len(has_cabin))

    space()

    print(survivors.pivot_table("Survived", index = "Sex", columns = "Pclass", aggfunc = "count"))

    space()

    print(victims.pivot_table("Survived", index = "Sex", columns = "Pclass", aggfunc = "count"))


def entropia():
    survival_groups = df.groupby(["Survived"]).count()
    totals = survival_groups.apply(sum)
    # print(totals)
    # print(len(df.PassengerId.unique()))
    df_totals = survival_groups._append(totals, ignore_index = True)
    survivors = df.Survived.value_counts()
    print(df_totals)
    print(df_totals.apply(prob_df))
    print("la entropía de Survived es de: " + per(entropy(prob(survivors[0], survivors[1], survivors.sum()), base = 2)))
    for column in df_totals:
        print("la entropía de " + column + " es del: " + per(entropy(prob(df_totals[column][0],df_totals[column][1], df_totals[column][2]), base = 2)))
    space()


df = pd.read_csv("data/train.csv")


