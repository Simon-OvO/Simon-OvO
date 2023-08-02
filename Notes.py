
#download dataset into browser 
{from pyodide.http import pyfetch

#使用异步函数（async）调取数据可以避免长时等待数据调取
#使用await表示等待异步函数完成后再执行下一步操作
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
}
{from js import fetch
import io

URL = ""
resp = await fetch(URL)
text = io.BytesIO((await resp.arrayBuffer()).to_py())#BytesIO读取字节文件,arraybuffer返回一个 promise 对象
df_can = pd.read_csv(text)
}

df = pd.read_csv（'',encoding = "ISO-8859-1"）#GB2312/GBK(汉字),unicode,UTF（兼容ISO-8859-1，比unicode简单）
airline_data =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})

#文本
import requests
data  = requests.get(url).text

import urllib
alice_novel = urllib.request.urlopen('')

#lib
data     'pandas'(df.drop_duplicates(subset=None,keep='first',inplace=False)#subset列名
                  df.info() #显示各列有多少非空值
                  df.isnull()#显示那个值为缺失值
                  df.dropna()
                  df.fillna(0)
                  series.value_counts()#计算值频率

         'scipy'#统计{stats.kstest(s['value'], 'norm', (u, std))
                      KstestResult(statistic=0.01441344628501079, pvalue=0.9855029319675546)，p值大于0.05为正太分布}
          numpy{linspace
                np.transpose()/ndarray.T
                numpy.linalg.inv()#倒数
                np.select#类似excel中xlookup）
                 condlist=[dnp[:,0]=='Yearly',dnp[:,0]=='Monthly',dnp[:,0]=='Weekly']
                 choicelist=[1,12,52]
                 df['NormalizedAnnualCompensation']=pd.DataFrame(np.select(condlist,choicelist)*dnp[:,1])


webscraping         from bs4 import BeautifulSoup  soup = BeautifulSoup(data,"html5lib")

                    for link in soup.find_all('a'):  # in html anchor/link is represented by the tag <a>
                         print(link.get('href'))#<a> 标签的 href 属性用于指定超链接目标的 URL
                        
                    for link in soup.find_all('img'):# in html image is represented by the tag <img>
                         print(link.get('src'))#<img> 标签的 src 属性是图像文件的 URL
                        
                    table = soup.find('table') # in html table is represented by the tag <table>
                    for row in table.find_all('tr'): # in html table row is represented by the tag <tr>
                        # Get all columns in each row.
                        cols = row.find_all('td') # in html a column is represented by the tag <td>
                        color_name = cols[2].getText() # store the value in column 3 as color_name
                        color_code = cols[3].getText() # store the value in column 4 as color_code
                        print("{}--->{}".format(color_name,color_code))
                        
analyse  'scikit-learn'{from sklearn.linear_model import LinearRegression
                        reg=LinearRegression()
                        reg.fit(x,y)
                        reg.coef_
                        reg.intercept_
                        reg.predict()
                        sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)#标准化多项式回归
                        sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)
}
         'skillsnetwork'
         'seaborn'
visualize'matplotlib'x和y：表示标签或者位置，用来指定显示的索引，默认为None
                     kind：表示绘图的类型，默认为line，折线图(line：折线图,bar/barh：柱状图（条形图），纵向/横向,pie：饼状图
                                              hist：直方图（数值频率分布）,box：箱型图,kde：密度图，主要对柱状图添加Kernel 概率密度线
                                              area：区域图（面积图）,scatter：散点图,hexbin：蜂巢图)
                     ax：子图，可以理解成第二坐标轴，默认None
                     subplots：是否对列分别作子图，默认False
                     sharex：共享x轴刻度、标签。如果ax为None，则默认为True，如果传入ax，则默认为False
                     sharey：共享y轴刻度、标签
                     layout：子图的行列布局，(rows, columns)
                     figsize：图形尺寸大小，(width, height)
                     use_index：用索引做x轴，默认True
                     title：图形的标题
                     grid：图形是否有网格，默认None
                     legend：子图的图例
                     style：对每列折线图设置线的类型，list or dict
                     logx：设置x轴刻度是否取对数，默认False
                     logy
                     loglog：同时设置x，y轴刻度是否取对数，默认False
                     xticks：设置x轴刻度值，序列形式（比如列表）
                     yticks
                     xlim：设置坐标轴的范围。数值，列表或元组（区间范围）
                     ylim
                     rot：轴标签（轴刻度）的显示旋转度数，默认None
                     fontsize : int, default None#设置轴刻度的字体大小
                     colormap：设置图的区域颜色
                     colorbar：柱子颜色
                     position：柱形图的对齐方式，取值范围[0,1]，默认0.5（中间对齐）
                     table：图下添加表，默认False。若为True，则使用DataFrame中的数据绘制表格
                     yerr：误差线
                     xerr
                     stacked：是否堆积，在折线图和柱状图中默认为False，在区域图中默认为True
                     sort_columns：对列名称进行排序，默认为False
                     secondary_y：设置第二个y轴（右辅助y轴），默认为False
                     mark_right : 当使用secondary_y轴时，在图例中自动用“(right)”标记列标签 ，默认True
                     x_compat：适配x轴刻度显示，默认为False。设置True可优化时间刻度的显示
                     1.2 其他常用说明
                       color：颜色
                       s：散点图大小，int类型
                       设置x,y轴名称
                      ax.set_ylabel(‘yyy’)
                      ax.set_xlabel(‘xxx’)
                     {df.plot(x=None, y=None, kind='line', ax=None, subplots=False, 
                              sharex=None, sharey=False, layout=None, figsize=None, use_index=True, title=None, grid=None, legend=True, 
                              style=None, logx=False, logy=False, loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, 
                              fontsize=None, colormap=None, position=0.5, table=False, yerr=None, xerr=None, stacked=True/False, sort_columns=False, 
                              secondary_y=False, mark_right=True, **kwds)
                      Line：plt.plot(x, y, color='red', linewidth=2)
                      Area：plt.fill_between(x, y1, y2, color='blue', alpha=0.5)
                      Histogram：plt.hist(data, bins=10, color='orange', edgecolor='black')
                      Bar：plt.bar(x, height, color='green', width=0.5)
                      Pie：plt.pie(sizes, labels=labels, colors=colors, explode=explode)
                      Box：plt.boxplot(data, notch=True)
                      Scatter：plt.scatter(x, y, color='purple', marker='o', s=50)
                      Subplotting：fig, axes = plt.subplots(nrows=2, ncols=2)
                      Customization: adding labels, title, legend, grid	Various customization	plt.title('Title')plt.xlabel('X Label')plt.ylabel('Y Label')plt.legend()plt.grid(True)
                                                                                
         'pandas'{Line：df.plot(x=’year’, y=’sales’, kind=’line’) 
                  Area：df.plot(kind='area')
                  Histogram：Series.plot.hist()
                  Bar：DataFrame.plot.bar()
                  Pie：s.plot(kind='pie’,autopct='%1.1f%%')
                  Box
                  Scatter Plot}
                  

         'seaborn'#countplot分类计数
         'Foleyum'#交互{# instantiate a feature group 
                        aus_reg = folium.map.FeatureGroup()
                        # Create a Folium map centered on Australia
                        Aus_map = folium.Map(location=[-25, 135], zoom_start=4)
                        # loop through the region and add to feature group
                        for lat, lng, lab in zip(reg.Lat, reg.Lon, reg.region):
                          aus_reg.add_child(
                            folium.features.CircleMarker(
                            [lat, lng],
                            popup=lab,
                            radius=5, # define how big you want the circle markers to be
                            color='red',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.6
                              )
                            )
                            }

# add incidents to map
Aus_map.add_child(aus_reg)}
         'Plotly'#交互{
                       
         }
         Ploty Express#简洁图表{px.scatter(dataframe, x=x_column, y=y_column，title='title')
                               px.line( x=x_column, y=y_column,'title')
                               px.bar( x=x_column, y=y_column,title='title',oritention=“h)
                               px.sunburst(dataframe, path=[col1,col2..], values='column',title='title')
                               px.histogram(x=x,title="title")
                               px.scatter(bub_data, x="City", y="Numberofcrimes",size="Numberofcrimes",hover_name="City", title='Crime Statistics')    #bubble
                               px.pie(values=x,names=y,title="title")
                               }
         Plotly Graph Objects{go.Scatter(x=x, y=y, mode='markers') #mode='lines'
                              fig.add_trace(go.Scatter(x=months_array, y=no_bicycle_sold_array))
                              fig.update_layout(title='Bicycle Sales', xaxis_title='Months', yaxis_title='Number of Bicycles Sold                     
                              }
         'PyWaffle'#比例可视化 
         'dash'#实时交互{dcc.Input(value='', type='text')
                        dcc.Graph(figure=fig)
                        html.Div(children=component_list,style={})
                        dcc.Dropdown(options=options_list, value=default_value
         }
#图表类型
pie#autopct='%1.1f%%'显示百分比,explode = [0,0,0,0,0.1]突出显示

#method
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork
import warnings
warnings.filterwarnings('ignore')
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output


interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))#互动（x,y,z）范围x-y,步数z

#function of plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.displot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.displot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    
#randomly split data into training and testing data using the function train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#tips
"* {} *".format()#字符串插入动态值
#regplot设置散点透明度scatter_kws={'alpha':1/3}
np.histogram#设置binedge
np.zeros(((height, width), dtype = np.uint))
matplotlib.pyplot.annotate#添加文本text：str  xy(float, float)
matplotlib.pyplot.gca（）#获取当前轴
np.arange([start, ]stop, [step, ]dtype=None, *, like=None)#设置间隔区间 np.arange(3,7,2)array([3, 5])
Axes.set_xticks(ticks, labels=None, *, minor=False, **kwargs)#minor次要刻度
Axes.grid(visible=None, which='major', axis='both', **kwargs)#网格线
html.Div([html.Div()，html.Div()],style={"display":'flex'})#flex表示并列排列
html.Br()#换行
df['Month'] = pd.to_datetime(df['Date']).dt.month#提取月份
df['Region'].unique()#唯一值，仅用于series
plt.xticks(ticks=[0, 1], labels=['Non-Recession', 'Recession'])
