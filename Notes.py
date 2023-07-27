
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
import urllib
alice_novel = urllib.request.urlopen('')

#lib
data     'pandas'(openpyxl(read excel))
         'scipy'
analyse  'scikit-learn'
         'skillsnetwork'
         'seaborn'
visualize'matplotlib'{Line：plt.plot(x, y, color='red', linewidth=2)
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
         'Foleyum'#交互
         'Plotly'#交互{
                       
         }
         Ploty Express#简洁图表{px.scatter(dataframe, x=x_column, y=y_column，title='title')
                               px.line( x=x_column, y=y_column,'title')
                               px.bar( x=x_column, y=y_column,title='title')
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
