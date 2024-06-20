import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standard_scale_train(X_train: pd.DataFrame, feature_names: list):
    """
    :param X_train: 훈련 데이터셋 (DataFrame)
    :param feature_names: 정규화를 수행할 특성들의 리스트
    :return: 정규화된 훈련 데이터셋, 훈련된 StandardScaler 객체
    """
    scaler = StandardScaler()
    train_data_scaled = X_train.copy()
    train_data_scaled[feature_names] = scaler.fit_transform(X_train[feature_names])
    return train_data_scaled, scaler


def standard_scale_val(X_val: pd.DataFrame, feature_names: list, scaler):
    """
    :param X_val: 검증 데이터셋 (DataFrame)
    :param feature_names: 정규화를 수행할 특성들의 리스트
    :param scaler: 훈련 데이터셋에 대해 훈련된 StandardScaler 객체
    :return: 정규화된 검증 데이터셋
    """
    X_val_scaled = X_val.copy()
    X_val_scaled[feature_names] = scaler.transform(X_val[feature_names])
    return X_val_scaled


def inverse_boxcox(y: np.array, lambda_: float):
    '''
    :param y: Boxcox변환된 강수량
    :param lambda_: Boxcox lambda
    :return: 원본 강수량
    '''
    if lambda_ == 0:
        return np.exp(y)
    else:
        return (np.exp(np.log(lambda_ * y + 1) / lambda_))
