from data_preprocessing import load_and_preprocess_data
from lightgbm_model import train_lightgbm
from stacking_model import train_stacking

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/tool_wear_data.csv')

    print("Training LightGBM model...")
    train_lightgbm(X_train, y_train, X_test, y_test)

    print("Training Stacking model...")
    train_stacking(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()