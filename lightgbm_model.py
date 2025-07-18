import lightgbm as lgb
from sklearn.metrics import classification_report

def train_lightgbm(X_train, y_train, X_test, y_test):
    params = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'objective': 'multiclass',
        'num_class': y_train.shape[1],
        'metric': 'multi_logloss'
    }

    train_data = lgb.Dataset(X_train, label=y_train.argmax(axis=1))
    test_data = lgb.Dataset(X_test, label=y_test.argmax(axis=1), reference=train_data)

    model = lgb.train(params, train_data, valid_sets=[test_data], early_stopping_rounds=10)
    preds = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), preds))
    return model