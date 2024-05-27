from cleaning import cols
from logistic_reggresion import accuracy as acc_log_reg
from decision_tree import accuracy as acc_dec_tree
from svm import accuracy as acc_svm
from naive_bayes import accuracy as acc_nai_bay
from EDA.quantitative_exploration import per
from statistics import mean

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
