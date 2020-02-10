from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from utils.utils import *


if __name__ == "__main__":
    # load config file from "config/config.json"
    config = load_config("config/config.json")

    # load dataset then preprocess and split into training and testing
    data_file = os.path.join("data", config["data"])    # dataset file (default: ./data/diabetes2_csv.csv)
    X, y = load_dataset(data_file, config["drop"], config["target"])    # load dataset using pandas
    X_prep = preprocess_data(X)  # then preprocess the loaded dataset
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=config["test_size"])  # split dataset

    # prepare the ml model using logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train.values.ravel())

    # predict the test dataset
    y_pred = model.predict(X_test)

    # save model after training
    save_dir = save_model(model)
    print("[*] model has been saved at ->", save_dir)

    # evaluate the score included
    #   * Confusion matrix -> (true_neg, false_pos, false_neg, true_pos)
    #   * Precision score
    #   * Recall score
    #   * F1 score
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test, y_pred).ravel()
    print("[*] from testing data", len(y_test), "rows.")

    print("")
    print("True Positive:", true_pos, "\t", "True Negative:", true_neg)
    print("False Positive:", false_pos, "\t", "False Negative:", false_neg)

    print("")
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Precision: {:.4f}".format(precision_score(y_test, y_pred)))
    print("Recall: {:.4f}".format(recall_score(y_test, y_pred)))
    print("F1 score: {:.4f}".format(f1_score(y_test, y_pred)))
