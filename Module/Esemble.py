import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class Esemble:
    def __init__(self, method, X_train, X_val, y_train, y_val, num_rounds, name):
        self.method = method
        self.X_train = X_train
        self.X_test = X_val
        self.y_train = y_train
        self.y_test = y_val
        self.num_rounds = num_rounds
        self.name = name
        self.unique_class = np.unique(y_val)

    def save_dict_to_txt(self, file_path, dict):
        with open(file_path, 'w') as f:
            for key, value in dict.items():
                f.write(f'{key}: {value}\n')

    def DecisionTree(self, params):
        bst = DecisionTreeClassifier(**params)
        bst.fit(self.X_train, self.y_train)
        y_pred = bst.predict(self.X_test)

        score = f1_score(self.y_test, y_pred, average='weighted')
        print(f'{score:.4f}')
        return score

    def lightGBM(self, params):
        bst = LGBMClassifier(**params, n_estimators=self.num_rounds)
        callbacks = [
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(100)
        ]
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], callbacks=callbacks)
        y_pred = bst.predict(self.X_test)

        score = f1_score(self.y_test, y_pred, average='weighted')
        print(f'{score:.4f}')
        return score

    def XGBoost(self, params):
        bst = XGBClassifier(**params, n_estimators=self.num_rounds)
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
        y_pred = bst.predict(self.X_test)

        score = f1_score(self.y_test, y_pred, average='weighted')
        print(f'{score:.4f}')
        return score

    def CatBoost(self, params):
        bst = CatBoostClassifier(**params, iterations=self.num_rounds)
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
        y_pred = bst.predict(self.X_test)

        score = f1_score(self.y_test, y_pred, average='weighted')
        print(f'{score:.4f}')
        return score

    def objective(self, trial):
        if self.method == 0:
            params = {
                'criterion': 'entropy',
            }
            accuracy = self.DecisionTree(params)

        if self.method == 1:
            params = {
                'device': 'cpu',
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': 10,
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
            }
            accuracy = self.lightGBM(params)

        if self.method == 3:
            params = {
                'tree_method': 'gpu_hist',
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 10,
                'booster': 'gbtree',
                'eta': 0.01,
                'early_stopping_rounds': 10,
            }
            accuracy = self.XGBoost(params)

        if self.method == 2:
            params = {
                'task_type': 'GPU',
                'objective': 'MultiClass',
                'eval_metric': 'MultiClass',
                'learning_rate': 0.01,
                'early_stopping_rounds': 10,
            }
            accuracy = self.CatBoost(params)

        return accuracy

    def save_best_model(self, best_params):
        if self.method == 0:
            best_params.update({
                'criterion': 'entropy',
            })

            bst = DecisionTreeClassifier(**best_params)
            bst.fit(self.X_train, self.y_train)
            joblib.dump(bst, f'File/DT/dt_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/DT/dt_{self.name}_params.txt', best_params)
            y_pred = bst.predict(self.X_test)
            score = f1_score(self.y_test, y_pred, average='weighted')
            print(f'{score:.4f}')
            print("Model saved!")

        if self.method == 1:
            best_params.update({
                'device': 'cpu',
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': 10,
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'early_stopping_round': 10,

            })

            bst = LGBMClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)])
            joblib.dump(bst, f'File/LGBM/lgb_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/LGBM/lgb_{self.name}_params.txt', best_params)
            y_pred = bst.predict(self.X_test)
            score = f1_score(self.y_test, y_pred, average='weighted')
            print(f'{score:.4f}')
            print("Model saved!")

        if self.method == 3:
            best_params.update({
                'tree_method': 'gpu_hist',
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 10,
                'booster': 'gbtree',
                'eta': 0.01,
                'early_stopping_rounds': 10,
            })

            bst = XGBClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
            joblib.dump(bst, f'File/XGB/xgb_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/XGB/xgb_{self.name}_params.txt', best_params)
            y_pred = bst.predict(self.X_test)
            score = f1_score(self.y_test, y_pred, average='weighted')
            print(f'{score:.4f}')
            print("Model saved!")

        if self.method == 2:
            best_params.update({
                'task_type': 'GPU',
                'objective': 'MultiClass',
                'eval_metric': 'MultiClass',
                'learning_rate': 0.01,
                'early_stopping_rounds': 10,
            })

            bst = CatBoostClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
            joblib.dump(bst, f'File/CAT/cat_{self.name}_model.pkl')
            self.save_dict_to_txt(f'File/CAT/cat_{self.name}_params.txt', best_params)
            y_pred = bst.predict(self.X_test)
            score = f1_score(self.y_test, y_pred, average='weighted')
            print(f'{score:.4f}')
            print("Model saved!")
