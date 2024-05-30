from cleaning import cols, X_test, y_test, class_names
import matplotlib.pyplot as plt
from logistic_reggresion import accuracy as acc_log_reg, lgr
from decision_tree import accuracy as acc_dec_tree, dtc
from svm import accuracy as acc_svm, svm
from naive_bayes import accuracy as acc_nai_bay, nb
from EDA.quantitative_exploration import per
import sklearn.metrics as metrics
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

def conf_mat(titles_options):
    fig , axs = plt.subplots(1, 4, figsize = (15, 4))
    i = 0
    np.set_printoptions(precision=2)
    for title, model, acc, color in titles_options:
        disp = metrics.ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels = class_names,
            cmap = plt.cm.Blues,
            normalize = "true",
            ax = axs[i], 
        )
        i += 1
        disp.ax_.set_title(title)

    plt.show()

def roc(titles_options, X_test, y_test):
    plt.title('Receiver Operating Characteristic Curve')
    plt.plot([0, 1], [0, 1],'r--')
    plt.legend(loc = 'lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    area_under_curves = []

    for title, model, acc, color in titles_options:

        if hasattr(model, 'predict_proba'): # check if the model can have a ROC curve to begin with
            pass
        else:
            continue

        probs = model.predict_proba(X_test)
        preds = probs[:,1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color, label = f'AUC of {title} = %0.2f' % roc_auc)

    plt.legend()
    plt.show()


titles_options = [
    ("Logistic Regression", lgr, acc_log_reg, 'b'),
    ("DecisionTreeClassifier", dtc, acc_dec_tree, 'g'),
    ("Support Vector Machine", svm, acc_svm, 'm'),
    ("Naive Bayes", nb, acc_nai_bay, 'y')
    ]



# conf_mat(titles_options)

roc(titles_options, X_test, y_test)

