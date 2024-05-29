from cleaning import cols, X_test, y_test, class_names
import matplotlib.pyplot as plt
from logistic_reggresion import accuracy as acc_log_reg, lr
from decision_tree import accuracy as acc_dec_tree, dtc
from svm import accuracy as acc_svm, svm
from naive_bayes import accuracy as acc_nai_bay, nb
from EDA.quantitative_exploration import per
from sklearn.metrics import ConfusionMatrixDisplay
from statistics import mean
import numpy as np

print("The columns used for the predictions are")
i = 1
for col in cols:
    print(f"{i}. {col}")
    i += 1

accuracies = [acc_dec_tree, acc_log_reg, acc_nai_bay, acc_svm]

print(f"\nlogistic regression accuracy: {per(acc_log_reg)}")
print(f"decision tree accuracy: {per(acc_dec_tree)}")
print(f"support vector machine accuracy: {per(acc_svm)}")
print(f"naive_bayes accuracy: {per(acc_nai_bay)}")
print(f"highest accuracy {per(max(accuracies))}")
print(f"mean accuracy: {per(mean(accuracies))}")

np.set_printoptions(precision=2)

fig , [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize = (15, 4))

titles_options = [
    ("Logistic Regression", lr, ax4, acc_log_reg),
    ("DecisionTreeClassifier", dtc, ax1, acc_dec_tree),
    ("Support Vector Machine", svm, ax2, acc_svm),
    ("Naive Bayes", nb, ax3, acc_nai_bay)
    ]

for title, model, ax, acc in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels = class_names,
        cmap = plt.cm.Blues,
        normalize = "true",
        ax = ax,
    )
    disp.ax_.set_title(title)

plt.show()