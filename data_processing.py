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

import urllib
alice_novel = urllib.request.urlopen('')}


{#从数据库获取数据
import sqlite3
conn = sqlite3.connect("m4_survey_data.sqlite") # open a database connection
df = pd.read_sql_query(QUERY,conn) #QUERY为要执行的sql语句
}

pandas.pivot_table()
#数据清洗
#空值(?/ /N/A)、重复值、异常值
df.drop_duplicates(subset=None,keep='first',inplace=False)#去除重复值，subset列名
df.info() #显示各列有多少非空值
df.isnull()#显示哪些值为缺失值
df.dropna()#去除空值
df.fillna(0)#替换空值
df.replace(missingvalue,newvalue)
series.value_counts()#计算值频率
(pandas.options.mode.use_inf_as_na = True#设置正负无穷为空值
df.isin()#可用于自定义空值判断)
df.astype(type) #当数据类型错误时转换数据类型,df.dtypes()查看数据类型

#增加新列，调用pandas，numpy
condlist=[dnp[:,0]=='Yearly',dnp[:,0]=='Monthly',dnp[:,0]=='Weekly']
choicelist=[1,12,52]
df['NormalizedAnnualCompensation']=pd.DataFrame(np.select(condlist,choicelist)*dnp[:,1])

#数据归一化
1.min-max:0~1
2.z-score:x=(x-Avg(x))/sigma标准差#标准差用df.std()计算

#分箱bining
bins=np.linspace(min(),max(),n)#返回n个等距数字
group_names=[]
df['binned']=pd.cut(df[],bins,labels=group_names,include_lowest=True)

#分类变量转为定量变量
pd.get_dummyies(df[])#生成数字类别，列名为类名，值为0，1


