#数据相关性研究需先进行正态分布检测，正态分布检测首选方法是图形观察，即利用直方图、P-P图或Q-Q图进行观察，其次，
#kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
#结果返回两个值：statistic → D值，pvalue → P值，p值大于0.05，为正态分布，
#H0:样本符合  H1:样本不符合 ，如果p>0.05接受H0 ,反之 
from scipy import stats
u = s['value'].mean()  # 计算均值
std = s['value'].std()  # 计算标准差MSE开方
stats.kstest(s['value'], 'norm', (u, std))

#计算列的两两相关性
df.corr(method='pearson', min_periods=1, numeric_only=False)
#method{‘pearson’, ‘kendall’, ‘spearman’} or callable返回correlation coefficient

#连续变量相关性检测，pearson correlation 
两个参数：
相关系数correlation coefficient：接近+/-1,强相关
P-value：0.001，0.05，0.1#确信度小于0.001强可信
pearson_coef,p_value=stats.pearsonr(df[],df[])

#分类变量相关性检测Chi-square卡方检测：检测分布是偶然的可能性
degree of freedom自由度=（row-1）*(column-1) p_value<0.05存在相关性
scipy.stats.chi2_contingency(cont_table,cprrection=True)#返回值：卡方测试值，p-value,自由度，期望值

#ANOVA: Analysis of Variance方差分析
方差分析（ANOVA）是一种统计方法，用于检验两个或多个组的均值之间是否存在显着差异。方差分析返回两个参数：
F 检验分数：计算实际均值与假设的偏差程度，并将其报告为 F 检验分数。分数越大意味着平均值之间的差异越大。
P 值：P 值表明我们计算的得分值在统计上的显着程度。
scipy.stats.f_oneway（*samples, axis=0）#sample1，sample1

#线性相关
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
y_hat=lm.predict(x_test)
lr.intercept_#b0截距
lr.coef_#b1斜率

decision_function(X)#Predict confidence scores for samples.
densify()#Convert coefficient matrix to dense array format.
fit(X, y[, sample_weight])#Fit the model according to the given training data.
get_metadata_routing()#Get metadata routing of this object.
get_params([deep])#Get parameters for this estimator.
predict(X)#Predict class labels for samples in X.
predict_log_proba(X)#Predict logarithm of probability estimates.
predict_proba(X)#Probability estimates.
score(X, y[, sample_weight])#R^2，X为测试组，y为真实数据
set_fit_request(*[, sample_weight])#Request metadata passed to the fit method.
set_params(**params)#Set the parameters of this estimator.
set_score_request(*[, sample_weight])#Request metadata passed to the score method.
sparsify()#Convert coefficient matrix to sparse format.

g=sns.PairGrid(df, hue="",vars=["", ""])#vars指定列
g=g.map(sns.scatterplot)
g = g.add_legend()  

g.map_offdiag(sns.scatterplot)  #指定非对角线绘图类型为散点图
g.map_diag(sns.histplot)  #指定对角线绘图类型为直方图
g = g.map_upper(sns.scatterplot)  # 指定上三角绘制散点图
g = g.map_lower(sns.kdeplot)  # 指定下三角绘制核密度图

g = sns.FacetGrid(tips, col="time",  row="sex")
g.map(sns.scatterplot, "total_bill", "tip")
g.refline(y=tips["tip"].median())#添加辅助线

#回归图
import seaborn as sns
sns.regplot(data=None, *, x=None, y=None, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, 
                           n_boot=1000, units=None, seed=None, order=1, logistic=False, lowess=False, robust=False, logx=False,
                           x_partial=None, y_partial=None, truncate=True, dropna=True, x_jitter=None, y_jitter=None, label=None, 
                           color=None, marker='o', scatter_kws=None, line_kws=None, ax=None)
sns.lmplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")#hue对于数据子集分别绘制回归图，以颜色区分
sns.lmplot(data=penguins, x="bill_length_mm", y="bill_depth_mm",col="species", row="sex", height=3,facet_kws=dict(sharex=False, sharey=False),
          )#允许xy轴在子图中不同
plt.ylim(0,)
#ci置信区间大小，回归线附近半透明区域，对于大量数据设置为0，避免计算
#order多阶
#残差图为对应x值的预测y值与实际值的差
sns.residplot(data=None, *, x=None, y=None, x_partial=None, y_partial=None, lowess=False, order=1, robust=False, dropna=True, label=None,
              color=None, scatter_kws=None, line_kws=None, ax=None)#lowess强调趋势 line_kws=dict(color="r")

#分布图
seaborn.displot(data=None, *, x=None, y=None, hue=None, row=None, col=None, weights=None, kind='hist', rug=False, rug_kws=None, 
                log_scale=None, legend=True, palette=None, hue_order=None, hue_norm=None, color=None, col_wrap=None, row_order=None, 
                col_order=None, height=5, aspect=1, facet_kws=None, **kwargs)
#kind="hist"/"kde"/"ecdf"（univariate-only)/
#kind=hist时，可增加kde曲线，kde=True
#绘制双变量图指定x，y 颜色深浅表述频率大小，仅适用于hist和kde sns.displot(data=penguins, x="flipper_length_mm", y="bill_length_mm")
 rug=True 在双变量绘图显示单变量观察结果
#其它关键字传递给底层函数，比如multiple="stack"
#可用height，aspect（纵横比）控制大小
g.set_axis_labels("Density (a.u.)", "Flipper length (mm)")
g.set_titles("{col_name} penguins")

#两图排列
import matploylib.plotly as plt
fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(12, 3))
sns.displot(df['price'], hist=False, color="r", label="Actual Value",ax=ax[])
#两图合一
ax1 = sns.displot(df['price'], hist=False, color="r", label="Actual Value")
sns.displot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
#将两个图画在一起需定位axis ax=sns.displot() /n sns.displot(,axis=ax) 

#多项式回归：首先需要标准化数据
#normalization
from sklearn.preprocessing import StandardScaler
scal=StandardScaler()
scal.fit(df[['','']])

#使用PolynomialFeatures将变量生成多项式比如 PolynomialFeatures(degree=2).fit([x1,x2],y)生成[x1,x2,x1*x2,x1^2,x2^2]
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True, order='C')
#degree:int or tuple (min_degree, max_degree)/order:F/C/include_bias：True为保留截距/interaction_only：True为仅产生交互项

#PolynomialFeatures对应method
fit(X[, y])#Compute number of output features.
fit_transform(X[, y])#Fit to data, then transform it.
get_feature_names_out([input_features])#Get output feature names for transformation.defult=None
get_metadata_routing()#Get metadata routing of this object.
get_params([deep])#Get parameters for this estimator.
set_output(*[, transform])#Set output container.
set_params(**params)#Set the parameters of this estimator.
transform(X) #Transform data to polynomial features.

#回归预测
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
x_new=np.linspace(min, max, 100)#生成多个数据
y_new = p(x_new)

from sklearn.pipeline import Pipeline
Input=[('scale',StandardScaler()),('polynormial',PolynomialFeatures(degree=2,)),('mode',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df[['','','','']],y)
y_hat=pipe.predict(df[['','','',']])


plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')#一个图多用'.'表示散点'-'表示折线'g-'字母表示颜色
