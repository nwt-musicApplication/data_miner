import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import webbrowser
if __name__ == '__main__':

    data_path = "data/listings.csv"   # 数据文件路径
    calendar = pd.read_csv(data_path)

    print(calendar.head())
    print(calendar.info())

    sns.set(rc={'figure.figsize': (19.7, 8.27)})    # 直方图展示缺失数据
    sns.heatmap(calendar.isnull(),
                yticklabels=False,
                cbar=False,
                cmap='viridis')
    plt.show()

    calendar['price'] = calendar['price'].str.replace(r'[$,]', '', regex=True).astype(np.float32)   # 数据类型转换
    calendar['neighbourhood_cleansed'] = calendar['neighbourhood_cleansed'].str.replace(r'[/,A-Z,a-z]', '', regex=True)

    sns.distplot(calendar["price"])
    plt.show()

    calendar1 = calendar[calendar.price < 10000]   # 数据价格小于10000的数据直方图绘制
    sns.distplot(calendar1["price"])
    plt.show()

    from matplotlib.font_manager import FontProperties     # 数据频数图绘制
    myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
    sns.set(font=myfont.get_name(), font_scale=0.5)
    sns.countplot(calendar["neighbourhood_cleansed"])
    plt.xticks(rotation=90)
    plt.show()

    sns.boxplot(x='neighbourhood_cleansed', y='price', data=calendar)   # 箱型图绘制
    plt.show()

    from matplotlib.font_manager import FontProperties    # 数据散点图绘制
    myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)
    sns.set(font=myfont.get_name(), font_scale=0.5)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(calendar.longitude, calendar.latitude, hue=calendar.neighbourhood_cleansed)
    plt.show()

    m = folium.Map([39.93, 116.40], zoom_start=11)     # 地图热图展示
    HeatMap(calendar[['latitude', 'longitude']].dropna(),
            radius=10,
            gradient={0.2: 'blue',
                      0.4: 'purple',
                      0.6: 'orange',
                      0.8: 'red'}
            ).add_to(m)
    file_path = r"./data/map.html"
    m.save(file_path)
    webbrowser.open(file_path)

    price_max = calendar['price'].max()     # 求五数概括
    price_min = calendar['price'].min()
    price_q1 = calendar['price'].quantile(0.25)
    price_q3 = calendar['price'].quantile(0.75)
    price_mid = calendar['price'].quantile(0.5)

    print(price_min)
    print(price_q1)
    print(price_mid)
    print(price_q3)
    print(price_max)


    f = calendar.plot(kind='scatter',
                      x='longitude',
                      y='latitude',
                      label='availability_30',
                      c='availability_365',
                      cmap=plt.get_cmap('jet'),
                      colorbar=True,
                      alpha=0.4)
    f.legend()
    plt.show()









