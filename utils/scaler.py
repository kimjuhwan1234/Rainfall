import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standard_scale_train(X_train: pd.DataFrame, feature_names: list):
    """
    :param X_train: 훈련 데이터셋 (DataFrame)
    :param feature_names: 정규화를 수행할 특성들의 리스트
    :return: 정규화된 훈련 데이터셋, 훈련된 StandardScaler 객체
    """
    scaler = MinMaxScaler()
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


def map_values(value: float):
    '''
    :param value: 데이터프레임 요소
    :return: 강수등급
    '''
    if value < 0.1:
        return 0
    elif 0.1 <= value < 0.2:
        return 1
    elif 0.2 <= value < 0.5:
        return 2
    elif 0.5 <= value < 1.0:
        return 3
    elif 1.0 <= value < 2.0:
        return 4
    elif 2.0 <= value < 5.0:
        return 5
    elif 5.0 <= value < 10.0:
        return 6
    elif 10.0 <= value < 20.0:
        return 7
    elif 20.0 <= value < 30.0:
        return 8
    elif value >= 30.0:
        return 9
    else:
        return np.nan  # 조건에 맞지 않는 값은 NaN으로 처리


def cal_CSI(csi_table: pd.DataFrame):
    '''
    :param csi_table: CSI 데이터프레임
    :return: CSI
    '''
    H = np.trace(csi_table.values) - csi_table.iloc[0, 0]
    F = csi_table.iloc[:, 1:].sum().sum() - H
    M = csi_table.iloc[1:, 0].sum()
    csi = H / (H + F + M)
    return csi
