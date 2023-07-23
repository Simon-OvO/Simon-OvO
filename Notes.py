
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
df_can = pd.read_csv（''）
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
         'Plotly'#交互
         'PyWaffle'#比例可视化 {#compute the proportion of each category with respect to the total
             total_values = df_dsn['*'].sum()
             category_proportions = df_dsn['*'] / total_values
             
             width = 40
             height = 10
             # compute the number of tiles for each category
             tiles_per_category = (category_proportions * total_num_tiles).round().astype(int)
             waffle_chart = np.zeros((height, width), dtype = np.uint)
             
             # define indices to loop through waffle chart
             category_index = 0
             tile_index = 0
             
             # populate the waffle chart
             for col in range(width):
                for row in range(height):
                   tile_index += 1

                   # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...
                   if tile_index > sum(tiles_per_category[0:category_index]):
                   # ...proceed to the next category
                       category_index += 1       
            
                   # set the class value to an integer, which increases with class
                   waffle_chart[row, col] = category_index
             
            print ('Waffle chart populated!')
            # instantiate a new figure object
            fig = plt.figure()
            # use matshow to display the waffle chart
            colormap = plt.cm.coolwarm
            plt.matshow(waffle_chart, cmap=colormap)
            plt.colorbar()
            plt.show()

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
