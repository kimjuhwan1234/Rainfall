import joblib
from sklearn.metrics import *
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor


class Esemble:
    def __init__(self, method, X_train, X_val, y_train, y_val, num_rounds, name):
        self.method = method
        self.X_train = X_train
        self.X_test = X_val
        self.y_train = y_train
        self.y_test = y_val
        self.num_rounds = num_rounds
        self.name = name

    def save_dict_to_txt(self, file_path, dict):
        with open(file_path, 'w') as f:
            for key, value in dict.items():
                f.write(f'{key}: {value}\n')

    def DecisionTree(self, params):
        bst = DecisionTreeRegressor(**params)
        bst.fit(self.X_train, self.y_train)
        y_pred = bst.predict(self.X_test)

        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        R_squre = r2_score(self.y_test, y_pred)
        print("DT R-square:", R_squre)
        return mape

    def lightGBM(self, params):
        bst = LGBMRegressor(**params, n_estimators=self.num_rounds, verbose_eval=100)
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)])
        y_pred = bst.predict(self.X_test)

        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        R_squre = r2_score(self.y_test, y_pred)
        print("LGBM R-square:", R_squre)
        return mape

    def XGBoost(self, params):
        bst = XGBRegressor(**params, n_estimators=self.num_rounds)
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
        y_pred = bst.predict(self.X_test)

        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        R_squre = r2_score(self.y_test, y_pred)
        print("XGB R-square:", R_squre)
        return mape

    def CatBoost(self, params):
        bst = CatBoostRegressor(**params, iterations=self.num_rounds)
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
        y_pred = bst.predict(self.X_test)

        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        R_squre = r2_score(self.y_test, y_pred)
        print("CAT R-square:", R_squre)
        return mape

    def objective(self, trial):
        if self.method == 0:
            params = {
                'criterion': 'friedman_mse',

                'max_features': trial.suggest_float('max_features', 0.1, 0.9),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
            }
            accuracy = self.DecisionTree(params)

        if self.method == 1:
            params = {
                'device': 'gpu',
                'objective': 'regression',
                'metric': 'mape',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'early_stopping_rounds': 10,

                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'lambda_l2': trial.suggest_float('reg_lambda', 1e-5, 10.0),

            }
            accuracy = self.lightGBM(params)

        if self.method == 2:
            params = {
                'tree_method': 'gpu_hist',
                'objective': 'reg:squarederror',
                'eval_metric': 'mape',
                'booster': 'gbtree',
                'eta': 0.01,
                'early_stopping_rounds': 10,

                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0),
            }
            accuracy = self.XGBoost(params)

        if self.method == 3:
            params = {
                'task_type': 'GPU',
                'objective': 'RMSE',
                'eval_metric': 'MAPE',
                'learning_rate': 0.01,
                'early_stopping_rounds': 10,

                'depth': trial.suggest_int('max_depth', 5, 20),
                'l2_leaf_reg': trial.suggest_float('reg_lambda', 1e-5, 10.0),
            }
            accuracy = self.CatBoost(params)

        return accuracy

    def save_best_model(self, best_params):
        if self.method == 0:
            best_params.update({
                'criterion': 'friedman_mse',
            })

            bst = DecisionTreeRegressor(**best_params)
            bst.fit(self.X_train, self.y_train)
            joblib.dump(bst, f'File/DT/dt_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/DT/dt_{self.name}_params.txt', best_params)
            print(f'{mean_absolute_percentage_error(self.y_test, bst.predict(self.X_test)):.4f}')
            print("Model saved!")

        if self.method == 1:
            best_params.update({
                'device': 'gpu',
                'objective': 'regression',
                'metric': 'mape',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'early_stopping_rounds': 10,

            })

            bst = LGBMRegressor(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)])
            joblib.dump(bst, f'File/LGBM/lgb_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/LGBM/lgb_{self.name}_params.txt', best_params)
            print(f'{mean_absolute_percentage_error(self.y_test, bst.predict(self.X_test)):.4f}')
            print("Model saved!")

        if self.method == 2:
            best_params.update({
                'tree_method': 'gpu_hist',
                'objective': 'reg:squarederror',
                'eval_metric': 'mape',
                'booster': 'gbtree',
                'eta': 0.01,
                'early_stopping_rounds': 10,
            })

            bst = XGBRegressor(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
            joblib.dump(bst, f'File/XGB/xgb_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/XGB/xgb_{self.name}_params.txt', best_params)
            print(f'{mean_absolute_percentage_error(self.y_test, bst.predict(self.X_test)):.4f}')
            print("Model saved!")

        if self.method == 3:
            best_params.update({
                'task_type': 'GPU',
                'objective': 'RMSE',
                'eval_metric': 'MAPE',
                'learning_rate': 0.01,
                'early_stopping_rounds': 10,
            })

            bst = CatBoostRegressor(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
            joblib.dump(bst, f'File/CAT/cat_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/CAT/cat_{self.name}_params.txt', best_params)
            print(f'{mean_absolute_percentage_error(self.y_test, bst.predict(self.X_test)):.4f}')
            print("Model saved!")
