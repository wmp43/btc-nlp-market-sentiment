from config import db_name, table2, clf_path
import sqlite3
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib
from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer


def db_ingestion(database_name, table_name):
    """
    Get Data From DB to process for Model
    """
    conn = sqlite3.connect(database_name)
    print('opened sqlite connection')

    query = f'SELECT * FROM {table_name}'

    df = pd.read_sql_query(query, conn)
    conn.close()
    print('closed sqlite connection')
    return df


def label_returns(df):
    std_returns = df['future_returns'].std()
    mean_returns = df['future_returns'].mean()

    conditions = [
        (df['future_returns'] > mean_returns + std_returns),
        (df['future_returns'] < mean_returns - std_returns),
    ]

    choices = ['pos_return', 'neg_return']

    df['return_class'] = np.select(conditions, choices, default='mod_return')

    le = LabelEncoder()
    df['return_class_encoded'] = le.fit_transform(df['return_class'])

    return df


def perform_clf_grid_search(df):
    X = df[['PCA1', 'PCA2', 'price', 'day_of_week', 'lagged_price', 'returns',
            'mean_neg_sentiment', 'max_neg_sentiment', 'min_neg_sentiment', 'std_neg_sentiment',
            'mean_pos_sentiment', 'max_pos_sentiment', 'min_pos_sentiment', 'std_pos_sentiment']][1:-1]
    y = df['return_class_encoded'][1:-1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [50, 100, 180, 250],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.03, 0.04, 0.05],
        'subsample': [0.2, 0.3, 0.4, 0.5],
        'colsample_bytree': [0.05, 0.06, 0.07, 0.1]
    }

    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, seed=42)

    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy',
                               cv=tscv,
                               verbose=2)

    grid_search.fit(X_train_resampled, y_train_resampled)

    print("Best parameters: ", grid_search.best_params_)
    print("Best accuracy on validation set: ", grid_search.best_score_)

    # Save the best model as a pickle file
    joblib.dump(grid_search.best_estimator_, '../models/xgb_clf_1.0.pkl')

    y_pred = grid_search.predict(X_val)
    print("Model Accuracy: ", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))


def clf_dev_master(database_name, table_name):
    df = db_ingestion(database_name, table_name)
    df = label_returns(df)
    perform_clf_grid_search(df)


# clf_dev_master(db_name, table_name)

def add_classifier_predictions(df, classifier_model_path):
    clf_model = load(classifier_model_path)

    X_clf = df[['PCA1', 'PCA2', 'price', 'day_of_week', 'lagged_price', 'returns',
                'mean_neg_sentiment', 'max_neg_sentiment', 'min_neg_sentiment', 'std_neg_sentiment',
                'mean_pos_sentiment', 'max_pos_sentiment', 'min_pos_sentiment', 'std_pos_sentiment']]

    new_data = X_clf.iloc[-1].values.reshape(1, -1)
    clf_predictions = clf_model.predict(new_data)

    # Create a copy of the last row and add the prediction
    last_row = df.iloc[-1].copy()
    last_row['clf_predictions'] = clf_predictions[0]

    return last_row


def perform_reg_grid_search(df, classifier_model_path):
    df = add_classifier_predictions(df, classifier_model_path)
    columns_to_drop = ['mean_sentiment_score_LABEL_0', 'max_sentiment_score_LABEL_0',
                       'min_sentiment_score_LABEL_0', 'std_sentiment_score_LABEL_0',
                       'mean_sentiment_score_LABEL_1', 'max_sentiment_score_LABEL_1',
                       'min_sentiment_score_LABEL_1', 'std_sentiment_score_LABEL_1',
                       'future_returns', 'time_published', 'index', 'titles', 'content']

    features = df.drop(columns=columns_to_drop)
    target = df['future_returns']

    n = len(df)

    split_idx = int(n * 0.9)

    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]

    param_grid = {
        'n_estimators': [100, 180, 250],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.04, 0.1],
        'subsample': [0.3, 0.5, 0.7],
        'colsample_bytree': [0.06, 0.1, 0.3],
        'reg_alpha': [0.1, 0.3, 0.5],
        'reg_lambda': [0.3, 0.6, 1.0]
    }

    model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric=['rmse'])

    tscv = TimeSeriesSplit(n_splits=3)

    # Define custom scoring metrics
    scoring = {'RMSE': 'neg_root_mean_squared_error',
               'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
               'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
               'Directional Accuracy': make_scorer(directional_accuracy, greater_is_better=True)}

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring=scoring,
                               cv=tscv,
                               verbose=1,
                               refit='MAPE')

    grid_search.fit(X_train, y_train)

    print("Best MAPE score on validation set: ", -grid_search.best_score_)
    print("Best RMSE score on validation set: ", -grid_search.cv_results_['mean_test_RMSE'][grid_search.best_index_])
    print("Best MAE score on validation set: ", -grid_search.cv_results_['mean_test_MAE'][grid_search.best_index_])
    print("Best Directional Accuracy on validation set: ",
          grid_search.cv_results_['mean_test_Directional Accuracy'][grid_search.best_index_])

    # Save the best model as a pickle file
    joblib.dump(grid_search.best_estimator_, '../models/xgb_reg_1.0.pkl')

    return grid_search.best_params_, -grid_search.best_score_, -grid_search.cv_results_['mean_test_MAPE'][
        grid_search.best_index_]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def directional_accuracy(y_true, y_pred):
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    return np.mean(direction_true == direction_pred) * 100


def evaluate_naive_approach(test_df):
    y_pred_naive = test_df['returns'][:-1]
    y_true_naive = test_df['future_returns'][1:]

    y_pred_naive = y_pred_naive.dropna()
    y_true_naive = y_true_naive.dropna()

    rmse_naive = mean_squared_error(y_true_naive, y_pred_naive, squared=False)
    mape_naive = mean_absolute_percentage_error(y_true_naive, y_pred_naive)

    print("RMSE - Naive Approach:", rmse_naive)
    print("MAPE - Naive Approach:", mape_naive)

    return rmse_naive, mape_naive


def evaluate_model(model, X_val, y_val):
    y_val = y_val.dropna()
    X_val = X_val.loc[y_val.index]

    y_pred = model.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)  # Added for MAE

    # Compute directional accuracy
    directional_accuracy = np.mean(np.sign(y_val) == np.sign(y_pred)) * 100

    print("RMSE - Model:", rmse)
    print("MAPE - Model:", mape)
    print("MAE - Model:", mae)  # Added for MAE
    print("Directional Accuracy - Model:", directional_accuracy)  # Added for Directional Accuracy

    return rmse, mape, mae, directional_accuracy


def reg_dev_master(clf_path, database_name, table_name):
    """
    Calls all previous functions to load clf, pred into new col
    train, GSCV to train and save reg model
    Evaluate the model

    Parameters
    ----------
    clf_path : .pkl
    """
    df = db_ingestion(database_name, table_name)
    df_with_clf_preds = add_classifier_predictions(df, clf_path)

    # Perform grid search for the regression model and get the best parameters
    best_params, best_rmse, best_mape = perform_reg_grid_search(df_with_clf_preds, clf_path)
    print("Best Regression Model Parameters:", best_params)

    # Split the data for evaluating the model
    n = len(df)
    split_idx = int(n * 0.9)
    X_val = df_with_clf_preds.drop(columns=['mean_sentiment_score_LABEL_0', 'max_sentiment_score_LABEL_0',
                                            'min_sentiment_score_LABEL_0', 'std_sentiment_score_LABEL_0',
                                            'mean_sentiment_score_LABEL_1', 'max_sentiment_score_LABEL_1',
                                            'min_sentiment_score_LABEL_1', 'std_sentiment_score_LABEL_1',
                                            'future_returns', 'time_published', 'index', 'titles',
                                            'content']).iloc[split_idx:]

    y_val = df_with_clf_preds['future_returns'].iloc[split_idx:]

    reg_model = joblib.load('../models/xgb_reg_1.0.pkl')

    rmse, mape, mae, da = evaluate_model(reg_model, X_val, y_val)

    return reg_model, rmse, mape, mae, da


#best_model, best_rmse, best_mape, best_mae, best_directional_accuracy = reg_dev_master(clf_path, db_name, table_name)
