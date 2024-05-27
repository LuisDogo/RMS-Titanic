import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def viz(df):
    fig, axs = plt.subplots(4, 2, layout="constrained")
    survived = df.Survived

    surv_count_class = df[df.Survived == 1].Pclass.value_counts()
    vict_count_class = df[df.Survived == 0].Pclass.value_counts()
    bar_class_surv = axs[0,0].bar(surv_count_class.index, surv_count_class)
    axs[0,0].set_title("Ticket Class of Survivors")
    bar_class_vict = axs[0,1].bar(vict_count_class.index, vict_count_class)
    axs[0,1].set_title("Ticket Class of Victims")

    surv_count_sex = df[df.Survived == 1].Sex.value_counts()
    vict_count_sex = df[df.Survived == 0].Sex.value_counts()
    bar_sex_surv = axs[1,0].bar(surv_count_sex.index, surv_count_sex)
    axs[1,0].set_title("Sex of Survivors")
    bar_sex_vict = axs[1,1].bar(vict_count_sex.index, vict_count_sex)
    axs[1,1].set_title("Sex of Victims")

    surv_age = df[df.Survived == 1].Age
    vict_age = df[df.Survived == 0].Age
    ages = [0,15,60,110]
    hist_age_surv = axs[2,0].hist(surv_age, bins = ages)
    axs[2,0].set_title("Age of Survivors")
    hist_age_vict = axs[2,1].hist(vict_age, bins = ages)
    axs[2,1].set_title("Age of Victims")

    surv_fare = df[df.Survived == 1].Fare
    vict_fare = df[df.Survived == 0].Fare
    hist_fare_surv = axs[3,0].hist(surv_fare, bins = 10)
    axs[3,0].set_title("Fare of Survivors")
    hist_fare_vict = axs[3,1].hist(vict_fare, bins = 10)
    axs[3,1].set_title("Fare of Victims")

    # surv_count_emb = df[df.Survived == 1].Embarked.value_counts()   # Quedaría mejor como bubble chart
    # vict_count_emb = df[df.Survived == 0].Embarked.value_counts()
    # # bubble_emb = axs[3, :].scatter(x = [surv_count_emb.index, vict_count_emb.index], y = ["Survived", "Victim"], size = [surv_count_emb, vict_count_emb])
    # lol = axs[3, ].plot(x = df.Age, y = df.Fare)
    # axs[3, ].set_title("Port of Embarkation of Survivors")

    fig.suptitle('Exploratory Data Analysis')

    plt.show()

viz(pd.read_csv("data/train.csv"))