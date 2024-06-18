import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_imabalance(df: pd.DataFrame, column_name: str):
    class_counts = df[column_name].value_counts()
    plt.pie(class_counts, labels=class_counts.index, startangle=140, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Credit Rating')
    plt.tight_layout()
    plt.show()


def plot_continuous_variable(df: pd.DataFrame, feature: str, kde: bool, log: bool):
    fig, ax = plt.subplots(figsize=(16, 7))

    desc_stats = df[feature].describe()

    if kde:
        sns.kdeplot(data=df, x=feature, shade=True, ax=ax)

    if not kde:
        sns.histplot(data=df, x=feature, ax=ax)

    else:
        sns.violinplot(data=df, x=feature, ax=ax)

    if log:
        ax.set_xscale('log')

    ax.set_title(df[feature].name + ' distribution', fontsize=15)
    ax.set_xlabel(df[feature].name, fontsize=15)

    stats_text = '\n'.join([f'{stat}: {value:.2f}' for stat, value in desc_stats.items()])
    ax.text(1, 1, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()