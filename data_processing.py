#数据获取

{#使用异步函数（async）调取数据可以避免长时等待数据调取
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
#空值、重复值、异常值
df.drop_duplicates(subset=None,keep='first',inplace=False)#去除重复值，subset列名
df.info() #显示各列有多少非空值
df.isnull()#显示哪些值为缺失值
df.dropna()#去除空值
df.fillna(0)#替换空值
series.value_counts()#计算值频率

#增加新列，调用pandas，numpy
condlist=[dnp[:,0]=='Yearly',dnp[:,0]=='Monthly',dnp[:,0]=='Weekly']
choicelist=[1,12,52]
df['NormalizedAnnualCompensation']=pd.DataFrame(np.select(condlist,choicelist)*dnp[:,1])


#数据相关性研究需先进行正态分布检测，正态分布检测首选方法是图形观察，即利用直方图、P-P图或Q-Q图进行观察，其次，
#kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
#结果返回两个值：statistic → D值，pvalue → P值，p值大于0.05，为正态分布，
#H0:样本符合  H1:样本不符合 ，如果p>0.05接受H0 ,反之 
from scipy import stats
u = s['value'].mean()  # 计算均值
std = s['value'].std()  # 计算标准差
stats.kstest(s['value'], 'norm', (u, std))
