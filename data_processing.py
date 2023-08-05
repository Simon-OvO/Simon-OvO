#数据获取

#download dataset into browser csv/json/excel/sql
{from pyodide.http import pyfetch

#使用异步函数（async）调取数据可以避免长时等待数据调取
#使用await表示等待异步函数完成后再执行下一步操作
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
}

df = pd.read_csv（'',encoding = "ISO-8859-1"）#GB2312/GBK(汉字),unicode,UTF（兼容ISO-8859-1，比unicode简单）
airline_data =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})
当没有列名时df.read_csv(path,encoding,header=None,names=column_names)

df.to_csv(path)#导出数据

#文本
import requests
data  = requests.get(url).text
(results = json.loads(data)#用json打开可以转成df格式
 df2 = pd.json_normalize(results)#将嵌套表格展开)

import urllib
alice_novel = urllib.request.urlopen('')}

#网页文件
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html5lib')
soup.prettify()# Returns formatted html
soup.find(tag)# Find the first instance of an HTML tag
soup.find_all(tag)# Find all instances of an HTML tag

table_bs.find_all(id="flight")
table_bs.find_all(string="Florida")
two_tables_bs.find("table",class_='pizza')

tag_object=soup.h3
tag_child =tag_object.b
tag_object.parent
tag_object.next_sibling#同级下一个
tag_child.attrs#直接返回字典
tag_child.string

for i,row in enumerate(table_rows):
    print("row",i)
    cells=row.find_all('td')
    for j,cell in enumerate(cells):
        print('colunm',j,"cell",cell)

for row in table.find_all('tr'): # in html table row is represented by the tag <tr>
    cols = row.find_all('td') # in html a column is represented by the tag <td>

{#从数据库获取数据
import sqlite3
conn = sqlite3.connect("m4_survey_data.sqlite") # open a database connection
df = pd.read_sql_query(QUERY,conn) #QUERY为要执行的sql语句

#操作数据库
cursor_obj = conn.cursor()
cursor_obj.execute("DROP TABLE IF EXISTS INSTRUCTOR")
conn.close()

%load_ext sql
%sql sqlite:///SQLiteMagic.db#连接数据库
dt=%sql select * from tablename
df=dt.DataFrame()
在前面加：使用python变量
}

    
df.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
         copy=None, indicator=False, validate=None)#how={‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}

pandas.pivot_tableDataFrame.pivot_table(values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, 
                                        dropna=True, margins_name='All', observed=False, sort=True)#透视表
df=df._get_numeric_data()



#数据清洗
#空值(?/ /N/A)、重复值、异常值
df.drop_duplicates(subset=None,keep='first',inplace=False)#去除重复值，subset列名
df.info() #显示各列有多少非空值
df.isnull()#布尔值表，值为True则为缺失值
df.dropna()#去除空值
df.fillna(0)#替换空值
df.replace(missingvalue,newvalue)
series.value_counts()#计算值频率 返回唯一行数
df[''].value_counts().idxmax()#计算最常见值
(pandas.options.mode.use_inf_as_na = True#设置正负无穷为空值
df.isin()#可用于自定义空值判断)
df.astype(type) #当数据类型错误时转换数据类型,参数copy默认值False，df.dtypes()查看数据类型
df.rename(columns={'':''}, inplace=True)#重命名列名

#自定义排序
df[].astype('category')
cat_dtype = CategoricalDtype(categories=[2, 1], ordered=True)
df[].astype(cat_dtype)

#增加新列，调用pandas，numpy
condlist=[dnp[:,0]=='Yearly',dnp[:,0]=='Monthly',dnp[:,0]=='Weekly']
choicelist=[1,12,52]
df['NormalizedAnnualCompensation']=pd.DataFrame(np.select(condlist,choicelist)*dnp[:,1])
#选择多列df1=df[['','','']]

#数据标准化
1.min-max:0~1
  sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), *, copy=True, clip=False)
2.z-score:x=(x-Avg(x))/sigma标准差（用df.std()计算）
  sklearn.preprocessing.StandardScaler（*, copy=True, with_mean=True, with_std=True）
3.减去中位数并根据四分位数范围缩放数据（默认为 IQR：四分位数范围）。 IQR 是第一个四分位数（0.25）和第三个四分位数（0.75）之间的范围。
  sklearn.preprocessing.RobustScaler(*, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True, 
                                     unit_variance=False)
4.幂变换是一系列参数化、单调变换，使数据更像高斯分布。这对于与异方差（非恒定方差）相关的建模问题或需要正态性的其他情况非常有用。
  sklearn.preprocessing.PowerTransformer(method='yeo-johnson', *, standardize=True, copy=True)
  #‘yeo-johnson’ 可用于正负值 ‘box-cox’只能用于正值
5.使用分位数信息转换特征。该方法将特征转换为服从均匀或正态分布。分散最频繁的值，减少异常值的影响，但是是非线性变换可能会扭曲线性相关性。
  sklearn.preprocessing.QuantileTransformer(*, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False, 
                                            subsample=10000, random_state=None, copy=True)
  #output_distribution：'uniform'均匀分布'normal'高斯分布

fit(X[, y])#Compute number of output features.
fit_transform(X[, y])#Fit to data, then transform it.#常用
get_feature_names_out([input_features])#Get output feature names for transformation.defult=None
get_metadata_routing()#Get metadata routing of this object.
get_params([deep])#Get parameters for this estimator.
set_output(*[, transform])#Set output container.
set_params(**params)#Set the parameters of this estimator.
transform(X) #Transform data to polynomial features.

#分箱bining
bins=np.linspace(min(),max(),n)#返回n个等距数字
group_names=[]
df['binned']=pd.cut(df[],bins,labels=group_names,include_lowest=True)

#分类变量转为定量变量
d_pd.get_dummyies(df[])#生成类别df，列名为类名，值为0，1
df = pd.concat([df, d_pd], axis=1)#连接表
df.drop("", axis = 1, inplace=True)#删除列

df.describe()#输出数据规格['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],默认不包含object类型，可用参数include
series.to_frame()
df.groupby(['',''],as_index=False)
df.get_group('')#获取分类数据
df.pivot(index='',columns='')#透视图的columns为两列的数据
MultiIndex.levels



