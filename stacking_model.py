from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train_stacking(X_train, y_train, X_test, y_test):
    base_learners = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier()),
        ('cart', DecisionTreeClassifier()),
        ('nb', GaussianNB()),
        ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier())
    ]
    final_estimator = LogisticRegression()
    model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator)

    param_grid = {
        'final_estimator__C': [0.1, 1, 10]
    }
    clf = GridSearchCV(model, param_grid, cv=3)
    clf.fit(X_train, y_train.argmax(axis=1))

    preds = clf.predict(X_test)
    print(classification_report(y_test.argmax(axis=1), preds))
    return clf