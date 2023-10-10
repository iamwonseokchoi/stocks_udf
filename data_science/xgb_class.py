from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import logging
import optuna
import sys

sys.path.append('../..')
from data_engineering.shared import read_parquet_from_s3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Log to file
file_handler = logging.FileHandler('../logs/xgboost.log')
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Log to terminal
stream_handler = logging.StreamHandler()
stream_format = logging.Formatter('%(message)s')
stream_handler.setFormatter(stream_format)
logger.addHandler(stream_handler)


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class XGBoostOptimizer:
    def __init__(self, X, y, ticker, trial_no, n_parallel, timestamp):
        self.X = X
        self.y = y
        self.ticker = ticker
        self.trial_no = trial_no
        self.n_parallel = n_parallel
        self.timestamp = timestamp
        self.pbar = tqdm(total=self.trial_no, desc='Hyperparameter Optimizing', position=0)
        self.best_model = None
        self.best_params = None

    def progress_bar(self, study, trial):
        self.pbar.update(1)

    def rolling_cv(self):
        window_size = 20
        for start in range(0, len(self.X) - 2 * window_size, window_size):
            yield (slice(start, start + window_size), slice(start + window_size, start + 2 * window_size))

    def objective(self, trial, alpha_bound=None):
        train_slice, valid_slice = next(self.rolling_cv())
        X_train, X_valid = self.X.iloc[train_slice], self.X.iloc[valid_slice]
        y_train, y_valid = self.y.iloc[train_slice], self.y.iloc[valid_slice]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
        }
        
        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['lambda'] = trial.suggest_float('lambda', 1e-15, 50.0, log=True)
            
            if alpha_bound:
                param['alpha'] = trial.suggest_float('alpha', alpha_bound[0], alpha_bound[1])
            else:
                param['alpha'] = trial.suggest_float('alpha', 1e-15, 50.0, log=True)
            
            param['subsample'] = trial.suggest_float('subsample', 0.1, 1.0)
            param['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.1, 1.0)
            param['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.1, 1.0)
            param['colsample_bynode'] = trial.suggest_float('colsample_bynode', 0.1, 1.0)
            param['max_depth'] = trial.suggest_int('max_depth', 1, 50)
            param['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 50)
            param['eta'] = trial.suggest_float('eta', 1e-15, 1.0, log=True)
            param['gamma'] = trial.suggest_float('gamma', 1e-15, 50.0, log=True)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
            param['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1e-6, 1000.0, log=True)
            param['tree_method'] = trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
            
            if param['booster'] == 'dart':
                param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                param['rate_drop'] = trial.suggest_float('rate_drop', 1e-15, 1.0, log=True)
                param['skip_drop'] = trial.suggest_float('skip_drop', 1e-15, 1.0, log=True)
        
        if param['booster'] == 'gblinear':
            param['lambda'] = trial.suggest_float('lambda', 1e-5, 5.0, log=True)
            
            if alpha_bound:
                param['alpha'] = trial.suggest_float('alpha', alpha_bound[0], alpha_bound[1])
            else:
                param['alpha'] = trial.suggest_float('alpha', 1e-5, 5.0, log=True)
            
            param['updater'] = trial.suggest_categorical('updater', ['shotgun', 'coord_descent']) 
            
            if param['booster'] == 'gblinear':
                param['feature_selector'] = trial.suggest_categorical('feature_selector', ['cyclic', 'shuffle'])

        bst = xgb.train(param, dtrain, evals=[(dvalid, 'validation')], verbose_eval=False, early_stopping_rounds=10)
        preds = bst.predict(dvalid)
        rmse = mean_squared_error(y_valid, preds, squared=False)

        return rmse

        bst = xgb.train(param, dtrain, evals=[(dvalid, 'validation')], verbose_eval=False, early_stopping_rounds=10)
        preds = bst.predict(dvalid)
        rmse = mean_squared_error(y_valid, preds, squared=False)

        return rmse

    def optimize(self, alpha_bound=None):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction='minimize')
        # Passing alpha_bound to the objective method here:
        study.optimize(lambda trial: self.objective(trial, alpha_bound=alpha_bound), 
                        n_trials=self.trial_no, 
                        callbacks=[self.progress_bar], 
                        n_jobs=self.n_parallel)
        self.pbar.close()

        best_params = study.best_params
        best_params['objective'] = 'reg:squarederror'
        best_params['eval_metric'] = 'rmse'
        
        dtrain = xgb.DMatrix(self.X, label=self.y)
        self.best_model = xgb.train(params=best_params, dtrain=dtrain, num_boost_round=study.best_trial.number)

        predictions = self.best_model.predict(dtrain)
        rmse_best = mean_squared_error(self.y, predictions, squared=False)
        mae_best = mean_absolute_error(self.y, predictions)
        r2_best = r2_score(self.y, predictions)
        median_absolute_error_best = median_absolute_error(self.y, predictions)

        logging.info('')
        logging.info('+'* 60)
        logging.info(f'[{self.ticker}-{self.timestamp}] XGBoost Best Model')
        logging.info('Number of finished trials:' + str(len(study.trials)))
        logging.info('+'* 60)
        logging.info('Best trial:')
        logging.info('-'* 60)
        trial = study.best_trial
        logging.info(f'RMSE Val: {trial.value}')
        logging.info(f'Best Model RMSE: {rmse_best}')
        logging.info(f'Best Model MAE: {mae_best}')
        logging.info(f'Best Model R2 Score: {r2_best}')
        logging.info(f'Best Model Median Absolute Error: {median_absolute_error_best}')
        logging.info('-'* 60)
        logging.info('Hyperparameters')
        logging.info('-'* 60)
        for key, value in trial.params.items():
            logging.info(f'    {key}: {value}')
        logging.info('='* 60)
        logging.info('')

    def save_best_model(self, save_path):
        dtrain = xgb.DMatrix(self.X, label=self.y)
        self.best_model = xgb.train(params=self.best_params, dtrain=dtrain)
        self.best_model.save_model(save_path)

    def plot_results(self, save_image_path):
        dnew = xgb.DMatrix(self.X)
        predictions = self.best_model.predict(dnew)
        
        true_values = self.y
        if len(true_values.shape) == 2:
            true_values = true_values.ravel()

        sns.set_style("whitegrid")
        plt.figure(figsize=(20, 10))

        blue = sns.color_palette("viridis", 10)[9]
        red = sns.color_palette("OrRd", 10)[9]
        
        sns.lineplot(x=self.X.index, y=true_values, label='Actual', color=blue, linewidth=2.5)
        sns.lineplot(x=self.X.index, y=predictions, label='XGBoost Predictions', color=red, linewidth=2.5)
        
        plt.xlabel('Time Index (Daily)', fontsize=16)
        plt.ylabel('Delta (Close-Open) Price', fontsize=16)
        plt.legend(frameon=True, loc='upper right', fontsize='medium')
        plt.title(f'Actual vs Predictions [XGBoost Best Model {self.ticker}-{self.timestamp}]', fontsize=20)
        
        sns.despine(left=True, bottom=True)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        plt.savefig(save_image_path, bbox_inches='tight')
        plt.show()


def featurize_parquet(parquet_path: str, target_col: str):
    df = read_parquet_from_s3(parquet_path)
    df.set_index('t', inplace=True)
    df_embedded = df['embedded_text'].apply(pd.Series)
    df_embedded = df_embedded.rename(columns = lambda x: 'embed_' + str(x))
    df = pd.concat([df, df_embedded], axis=1)
    df.drop('embedded_text', axis=1, inplace=True)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y