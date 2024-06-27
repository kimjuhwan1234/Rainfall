import os
import optuna
import warnings
import pandas as pd
from Module.Esemble import *

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    GMM_list = ['GMM0', 'GMM2', 'GMM4', 'GMM6', 'GMM7', 'GMM8', 'GMM9', 'GMM10', 'GMM11', 'GMM14']
    directory = 'Database/train/'
    files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    X_val = pd.read_csv('Database/val/X_val_norm.csv', index_col=0)
    y_val = pd.read_csv('Database/val/y_val.csv', index_col=0)

    for i, code in enumerate(GMM_list):
        file_list = [sentence for sentence in files if code in sentence]
        X_train = pd.read_csv(os.path.join(directory, file_list[0]), index_col=0)
        y_train = pd.read_csv(os.path.join(directory, file_list[1]), index_col=0)

        for i in range(2, 4):
            # {DT=0, lightGBM=1, CatBoost=2, XGBoost=3}
            E = Esemble(i, X_train, X_val, y_train, y_val, 1000, code)

            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(E.objective, n_trials=3)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)
            E.save_best_model(study.best_trial.params)
