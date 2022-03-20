import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def data_ana():
    file_path = 'data/output.csv'
    data_read = pd.read_csv(file_path)

    imputer = KNNImputer(n_neighbors=1)
    imputer.fit_transform(data_read)
    sns.distplot(data_read[data_read.price < 500]['price'])
    plt.show()


def data_quantize(calendar):  # 数据数值化
    country_key = list(set(calendar['country']))
    region_1_key = list(set(calendar['region_1']))
    region_2_key = list(set(calendar['region_2']))
    variety_key = list(set(calendar['variety']))
    winery_key = list(set(calendar['winery']))

    price_list = calendar['price']
    country_list = []
    region_1_list = []
    region_2_list = []
    variety_list = []
    winery_list = []
    for i in range(len(calendar)):
        country_list.append(country_key.index(calendar['country'][i]))
        region_1_list.append(region_1_key.index(calendar['region_1'][i]))
        region_2_list.append(region_2_key.index(calendar['region_2'][i]))
        variety_list.append(variety_key.index(calendar['variety'][i]))
        winery_list.append(winery_key.index(calendar['winery'][i]))

    calendar_tmp = pd.DataFrame({'price': price_list,
                                 'country': country_list,
                                 'region_1': region_1_list,
                                 'region_2': region_2_list,
                                 'variety': variety_list,
                                 'winery': variety_list})
    write_path = 'data/output.csv'
    calendar_tmp.to_csv(write_path)


if __name__ == '__main__':

    data_path = "data/winemag-data-130k-v2.csv"   # 数据读取路径
    calendar = pd.read_csv(data_path)

    print(calendar.head())
    print(calendar.info())

    sns.set(rc={'figure.figsize': (19.7, 8.27)})
    sns.heatmap(calendar.isnull(),
                yticklabels=False,
                cbar=False,
                cmap='viridis')
    plt.show()

    calendar['price'] = calendar['price'].astype(np.float64)

    price_none_count = calendar['price'].isnull().sum()   # 空值计数统计

    price_value_count = calendar['price'].value_counts(sort=True)  # 频数计数统计

    price_max = calendar['price'].max()
    price_min = calendar['price'].min()
    price_q1 = calendar['price'].quantile(0.25)
    price_q3 = calendar['price'].quantile(0.75)
    price_mid = calendar['price'].quantile(0.5)
    print(price_min)
    print(price_q1)
    print(price_mid)
    print(price_q3)
    print(price_max)

    price_list = np.unique(calendar['price'])   # price_key统计

    sns.distplot(calendar["price"])    # price直方图
    calendar1 = calendar[calendar.price < 500]
    sns.distplot(calendar1["price"])
    plt.show()

    sns.boxplot(y='price', data=calendar)  # price箱型图
    plt.show()

    new_calendar = calendar.dropna(subset=['price'])    # 删除缺失值
    sns.distplot(new_calendar["price"])
    plt.show()

    calendar['price'] = calendar['price'].fillna(calendar['price'].mode())  # 众数填充
    sns.distplot(calendar["price"])
    plt.show()

    plt.figure(figsize=(12,12))   # price_points散点图
    sns.scatterplot(y='price', x='points', data=calendar)
    plt.show()

    a = calendar[['points', 'price']].corr(method='pearson')  # spearman系数计算数据相关性
    print('价格和评分的整体相关性系数为%.4f' % (a[0:1]['price']))

    from sklearn import linear_model    # 计算线性方程
    x = np.array(calendar[calendar.price < 3300.0]['price']).reshape(-1, 1)
    y = np.array(calendar[calendar.price < 3300.0]['points']).reshape(-1, 1)
    model = linear_model.LinearRegression()
    model.fit(x, y)
    coef = model.coef_[0][0]
    model_intercept = model.intercept_[0]
    print('线性回归方程为：' + '\n' + 'y={}*x+{}'.format(coef, model_intercept))

    price_tmp = calendar[['price', 'points']]   # 以线性函数补充缺失值，绘图
    for i in range(len(price_tmp)):
        if np.isnan(price_tmp['price'][i]):
            price_tmp['price'][i] = coef * price_tmp['points'][i]
    price_tmp = price_tmp[price_tmp.price < 500]
    sns.distplot(price_tmp["price"])
    plt.show()

    plt.figure(figsize=(12, 12))    # price<500的数据分析图
    sns.lmplot(x='price', y='points', data=(calendar[calendar.price < 500][['points', 'price']]))
    plt.show()

    data_quantize(calendar)  # 数据数值化

    data_ana()




