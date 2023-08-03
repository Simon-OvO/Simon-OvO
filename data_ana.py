#数据相关性研究需先进行正态分布检测，正态分布检测首选方法是图形观察，即利用直方图、P-P图或Q-Q图进行观察，其次，
#kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
#结果返回两个值：statistic → D值，pvalue → P值，p值大于0.05，为正态分布，
#H0:样本符合  H1:样本不符合 ，如果p>0.05接受H0 ,反之 
from scipy import stats
u = s['value'].mean()  # 计算均值
std = s['value'].std()  # 计算标准差
stats.kstest(s['value'], 'norm', (u, std))

#连续变量相关性检测，pearson correlation 
两个参数：
相关系数correlation coefficient：接近+/-1,强相关
P-value：0.001，0.05，0.1确信度
pearson_coef,p_value=stats.pearsonr(df[],df[])

#分类变量相关性检测Chi-square卡方检测：检测分布是偶然的可能性
degree of freedom自由度=（row-1）*(column-1) p_value<0.05存在相关性
scipy.stats.chi2_contingency(cont_table,cprrection=True)#返回值：卡方测试值，p-value,自由度，期望值
