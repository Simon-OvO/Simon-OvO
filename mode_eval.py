Mean Squared Error(MSE)平均平方差
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)

R-squared 决定系数
R^2=1-MSE(回归)/MSE(均值或者其它)，接近1则表明模型适用 lm.score(x,yhat)
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)

import numpy as np
new_input=np.arrange(1,101,1).reshape(-1,1)#reshape将数据改为一列
yhat=lm.predict(new_input)
#数据分区
from sklearn.model_selection import trian_test_split
x_train,x_test,y_train,y_test=trian_test_split(x_data,y_data,test_size=0.3,random_state=0)
#分区交叉测试
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,x,y,cv=3)#cv：分区model：训练模型
#分区交叉预测
from sklearn.model_selection import cross_val_predict
yhat=cross_val_predict(model,x,y,cv=3)

from sklearn.linear_model import Ridge
rm=Ridgesklearn.linear_model.Ridge(alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', 
                                   positive=False, random_state=None)#alpha:  
rm.fit(X,y)
yhat=rm.predict(X)

#筛选合适的参数值，默认r^2筛选
from sklearn.model_selection import GridSearchCV
parameters=[{'alpha':[1,10,100,1000],'normalize':[True,False]}]
rg=Ridge()
gs=GridSearchCV(rg,parameters,cv=4)
gs.fit（x，y）
gs.best_estimator_
scores=gs.cv_results_
scores['mean_test_score']

#筛选合适的阶数
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)  
    x_train_pr = pr.fit_transform(x_train[['horsepower']])    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])        
    lr.fit(x_train_pr, y_train)    
    Rsqu_test.append(lr.score(x_test_pr, y_test))
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')

#画出两组数据的kde图比如：预测值和真实值，判断数据是否拟合，适用于多元关系
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction,color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction,color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

#画出训练值，测试值，以及回归函数
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

#画出指定项数和测试size的训练值，测试值，以及回归函数
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train,y_test, poly, pr)

#交互板
from ipywidgets import interact, interactive, fixed, interact_manual
interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))


from tqdm import tqdm
Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = np.array(range(0,1000))#range返回范围内数字，python内置函数，range(start, stop[, step])
pbar = tqdm(Alpha)#进度条
for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})#输入字典，显示指标

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

width = 6
height = 3
plt.figure(figsize=(width, height))
plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.xlim(0,)
plt.legend()
