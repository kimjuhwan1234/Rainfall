import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler


def plot_imabalance(df: pd.DataFrame, column_name: str):
    '''
    :param df: 확인할 데이터프레임
    :param column_name: 불균형을 확인할 이산형변수 칼럼이름
    :return: 파이그래프 출력
    '''
    class_counts = df[column_name].value_counts()
    plt.pie(class_counts, labels=class_counts.index, startangle=140, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Credit Rating')
    plt.tight_layout()
    plt.show()


def plot_continuous_variable(df: pd.DataFrame, feature: str, method: str, log: bool, scaler: bool):
    '''
    :param df: 확인할 데이터프레임
    :param feature: 확인할 연속형 변수의 칼럼이름
    :param method: kde = 연속함수, hist = 히스토그램, desity = 밀도
    :param log: log변환여부
    :param scaler: 표준화여부
    :return: method에서 지정한 그래프출력
    '''
    fig, ax = plt.subplots(figsize=(16, 7))
    data = df[feature]

    if log:
        # df[feature]=pd.DataFrame(df[feature]).applymap(lambda x: np.log(x+1))
        data = pd.Series(stats.boxcox(df[feature] + 0.0000001)[0])

    if scaler:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        data = pd.DataFrame(data=scaled_data, columns=pd.DataFrame(data).columns, index=df.index)

    desc_stats = pd.DataFrame(data).describe()

    if method == 'kde':
        sns.kdeplot(data=data, shade=True, ax=ax)

    if method == 'hist':
        sns.histplot(data=data, ax=ax)

    if method == 'density':
        sns.violinplot(data=data, ax=ax)

    ax.set_title(df[feature].name + ' distribution', fontsize=15)
    ax.set_xlabel(df[feature].name, fontsize=15)

    stats_text = ''
    for column in desc_stats.columns:
        stats_text += f'\nColumn: {column}\n'
        stats_text += '\n'.join([f'{stat}: {value:.2f}' for stat, value in desc_stats[column].items()])
    ax.text(1, 1, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
