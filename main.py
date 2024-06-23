import os
import optuna
import warnings
import pandas as pd
from Module.Esemble import *

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    STN_list = ['STN001', 'STN002', 'STN003', 'STN004', 'STN005', 'STN006', 'STN007', 'STN008', 'STN009', 'STN010',
                'STN011', 'STN012', 'STN013', 'STN014', 'STN015', 'STN016', 'STN017', 'STN018', 'STN019', 'STN020']

    GMM_list = [f'GMM{i}' for i in range(14)]
    directory = 'Database/train/'
    files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    X_val = pd.read_csv('Database/val/X_val_norm.csv', index_col=0)
    y_val = pd.read_csv('Database/val/y_val.csv', index_col=0)

    for i, code in enumerate(GMM_list):
        file_list = [sentence for sentence in files if code in sentence]
        X_train = pd.read_csv(os.path.join(directory, file_list[0]), index_col=0)
        y_train = pd.read_csv(os.path.join(directory, file_list[1]), index_col=0)

        for i in range(0, 4):
            # {DT=0, lightGBM=1, XGBoost=2, CatBoost=3}
            E = Esemble(i, X_train, X_val, y_train, y_val, 1000, code)

            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)
            E.save_best_model(study.best_trial.params)
