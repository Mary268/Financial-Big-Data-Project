import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import itertools
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import roc_curve
from pylab import scatter, legend ,show, xlabel, ylabel
import matplotlib.image as mpimg

newEDA = 0
newPlots = 0
n_sFeatures = 10

#----------------- Step 1: Data initialize ------------------------------------
print('---------- Step 1: Data initialize -----------------------------------')
Loan17Q1=pd.read_csv('Loan17Q1.csv')
Loan17Q1 = Loan17Q1[Loan17Q1.application_type=='Individual']
Loan17Q1=Loan17Q1[Loan17Q1.hardship_flag=='N']
Loan17Q1=Loan17Q1[Loan17Q1.debt_settlement_flag=='N']
Loan17Q1.reset_index(drop=True,inplace=True)


#----------------- Step 2: Data cleaning, pre-processing and data balancing ---
print('--------- Step 2: Data cleaning, pre-processing and data balancing ---')
hardship_cols=range(129,143)#not include 139
Loan17Q1.drop(Loan17Q1.columns[hardship_cols], axis=1, inplace=True)

# ----- Drop duplicated columns: Start
j=0
jlist=[]
while j<137:
    j=j+1
    if Loan17Q1.duplicated(subset=Loan17Q1.columns[j-1]).sum() == 91929:
        jlist.append(j-1)
    else:
        continue

Loan17Q1.drop(Loan17Q1.columns[jlist], 1, inplace=True)
droplist=['url','id']
Loan17Q1.drop(droplist,1,inplace=True)
# ----- Drop duplicated columns: End

# ----- Pre-process data: Type conversion and Cleaning NAN/NULL
# Converting percentage to float
Loan17Q1['int_rate']=Loan17Q1.int_rate.str.strip("%").astype(float)
Loan17Q1['revol_util']=Loan17Q1.revol_util.str.strip("%").astype(float)
Loan17Q1.fillna(0,inplace=True)
# Targer Variable 'loan_status': add conversion column
Loan17Q1['loan_status_str'] = Loan17Q1['loan_status'] 
Loan17Q1['loan_status'] = 0
# update 'loan_status' = 1 if they charged off; otherwise = 0 
Loan17Q1.loc[Loan17Q1['loan_status_str'] =='Charged Off', 'loan_status'] = 1

# Print Initial Data Shape
print('Data Shape before balancing: Charged-off Not-Charged-off Ratio')
print('                             '+ str(sum(Loan17Q1['loan_status'])) + '        '+
      str(len(Loan17Q1['loan_status']) - sum(Loan17Q1['loan_status'])) + '          '+
      str(sum(Loan17Q1['loan_status'])/(len(Loan17Q1['loan_status'])-sum(Loan17Q1['loan_status']))) )
'''
Data Shape before balancing: Charged-off Not-Charged-off Ratio
                             4578        87352          0.05240864548035534
'''
# Remove conversion column
Loan17Q1=Loan17Q1.drop('loan_status_str',axis=1)
Loan17Q1=Loan17Q1.drop('out_prncp_inv',axis=1)
Loan17Q1=Loan17Q1.drop('out_prncp',axis=1)
# Potential Explanatory Varibales (colunms)
variables = list(Loan17Q1.columns.values)

# ----- Balance Sampling / UnderSampling use all 'loan_status==1' data; 
# But only use 10% of 'loan_status==0'
TrueP17Q1=Loan17Q1[Loan17Q1.loan_status==1]
TrueN17Q1=Loan17Q1[Loan17Q1.loan_status==0].sample(frac=0.1,random_state=99)

# Merge all 2:1 samples of data together
data17Q1=pd.concat([TrueP17Q1,TrueN17Q1])

print('Data Shape after balancing: Charged-off Not-Charged-off Ratio')
print('                             '+ str(sum(data17Q1['loan_status'])) + '        '+
      str(len(data17Q1['loan_status']) - sum(data17Q1['loan_status'])) + '            '+
      str(sum(data17Q1['loan_status'])/(len(data17Q1['loan_status'])-sum(data17Q1['loan_status']))) )
'''
Data Shape after balancing: Charged-off Not-Charged-off Ratio
                             4578        8735          0.5240984544934173
'''
# Distribution of 0/1 after balance: 8735/4578 = 2:1
# plot for Before Sampling vs After Sampling
prob_1_after = len(TrueP17Q1)/(len(TrueP17Q1)+len(TrueN17Q1))
prob_1_before = len(TrueP17Q1)/len(Loan17Q1)

pievalues_1=[prob_1_before,1-prob_1_before]
labels_1=['Charged Off','Not Charged Off']
colors_1=['blue', 'lightblue']
plt.pie(pievalues_1, labels=labels_1, colors=colors_1, startangle=90,explode=[0.1,0],autopct='%.2f%%')
plt.title('Charged Off Rate Before Sampling')
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()

pievalues_1=[prob_1_after,1-prob_1_after]
labels_1=['Charged Off','Not Charged Off']
colors_1=['blue', 'lightblue']
plt.pie(pievalues_1, labels=labels_1, colors=colors_1, startangle=90,explode=[0.1,0],autopct='%.2f%%')
plt.title('Charged Off Rate After Sampling')
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()

# Storage Target Varibale: array Status17Q1
Status17Q1 = (data17Q1.loc[:,'loan_status'].values).reshape(len(data17Q1.loan_status),1)
# Target Varibale: dataframe Y
Y = data17Q1['loan_status']
data_new=pd.DataFrame()
data_new=data17Q1.drop('loan_status',axis=1)

# Convert categorial value to Dummy variables for further modeling
data_new2=pd.get_dummies(data_new)


#---------------------   Step 3: Feature selection   --------------------------
print('---------   Step 3: Feature selection   ------------------------------')
#--------------------- Method 1: Univariate feature selection -----------------
selector = SelectKBest(chi2, k = n_sFeatures)
X_new = selector.fit_transform(data_new2, Status17Q1)
# Get name of selected variables
names = data_new2.columns.values[selector.get_support()]
# Get F_Score of selected variables
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores'], ascending = [False])
print(ns_df_sorted)
'''
                Feat_names      F_Scores
3          total_rec_prncp  1.461233e+07
9          tot_hi_cred_lim  1.217372e+07
1              total_pymnt  9.102228e+06
2          total_pymnt_inv  9.100234e+06
6          last_pymnt_amnt  8.822997e+06
7              tot_cur_bal  8.648178e+06
4               recoveries  7.256256e+06
0               annual_inc  1.846053e+06
8           bc_open_to_buy  1.595737e+06
5  collection_recovery_fee  1.255572e+06
'''
# ------- Store Explanatory Data Analysis Plots: Start --------
#use 10 features selected from Univariate feature selection
data_sFeatures=pd.DataFrame()
# using Feature selection: names
# not using Feature selection: data_new.columns
for name in names:
    data_sFeatures[name]=data_new[name]

Y_DT = pd.DataFrame()
Y_DT['loan_status'] = Y

# -----  When need generate EDA Plots (newEDA==1), Run this parts
if newEDA==1:
    i = n_sFeatures
    for name in ns_df_sorted['Feat_names']:
        plt.scatter(data17Q1[name],Y, color='purple',s=10)
        plt.title(name+' variable vs Charge-off Label')
        plt.xlabel(name)
        plt.ylabel("Charge-off")
        plt.savefig('EDA/'+str(i)+' '+name+' vs Label.png')
        i = i-1
    
    pd.get_dummies(data_sFeatures).corr().to_csv('correlations of variables.csv')
    
    i = n_sFeatures
    for name in names:
        plt.hist(data_sFeatures[name],color='purple')
        plt.title(name+' variable Distribution')
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.savefig('EDA/hist'+str(i)+' '+name+'.png')
        i = i-1
#  ------ When need generate EDA Plots (newEDA==1), Run this parts

#--------------------- Method 2: Desicion Tree feature selection -------------- 
# Desicion Tree for Feature Selection; Input Data contains all features
DT_FS = DecisionTreeClassifier(criterion="entropy", min_samples_leaf = 50)
DT_FS.fit(data_new2, Y)
# Visulize Desicion Tree
dot_data = StringIO()
feature_names= data_new2.columns
class_names=['Not Charged Off','Charged Off']
export_graphviz(DT_FS, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names=feature_names,class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
img= Image(graph.create_png())
graph.write_png("DT_allData_allFeature.png")

# ------------------ Split Training and Testing data for futher modeling ------
X_train, X_test, Y_train, Y_test = train_test_split(
        data_sFeatures, Y_DT, test_size=0.3, random_state=42)

#-----------------------   Step 4: Classification Analysis  -------------------
print('---------   Step 4: Classification Analysis   ------------------------')
#-----------------------   Method 1: Decision Tree   --------------------------
# Input Data contains the selected k best features
DT_train = DecisionTreeClassifier(criterion="entropy", min_samples_leaf = 150)
# Train the DecisionTree using the training sets
DT_train.fit(X_train, Y_train['loan_status'])

#----- Cross-Validation on K=10 folds: Training Accuracy
# 10 folds cross validation on Training data
kf = KFold(len(X_train), n_folds=10)
scores = cross_val_score(DT_train, X_train, Y_train['loan_status'], scoring = 'accuracy', cv=kf)
print('Desicion Tree 10 folds cross-validation Average Accuracy:')
print(scores)
print(scores.mean())
'''
0.9086826892491807
'''

#----- Evaluation: Testing Accuracy
# Confusion Matrix of Desicion Tree Binary Classifier in Testing Data
print('Desicion Tree on test data set accuracy:')
DT_Y_test=DT_train.predict(X_test)
print(DT_Y_test)
print('Desicion Tree on test dataset Confusion Matrix:')
cm = metrics.confusion_matrix(Y_test, DT_Y_test, labels=[0,1])
print('            y_pred\n            0    1')
print('y_true 0   '+ '  ' . join(str(a) for a in cm[0]) + '  specificity:  %f'% (cm[0][0]/sum(cm[0])) )
print('       1   '+ '  ' . join(str(a) for a in cm[1]) + '  sensitivity:  %f'% (cm[1][1]/sum(cm[1])) )
'''
		         y_pred
            0    1
y_true 0   2397  196  specificity:  0.924412
       1   202  1199  sensitivity:  0.855817
'''

# Visulize Desicion Tree Classification Model on all data
DT17 = DecisionTreeClassifier(criterion="entropy", min_samples_leaf = 150)
DT17.fit(data_sFeatures, Y)
dot_data = StringIO()
feature_names= X_train.columns
class_names=['Not Charged Off','Charged Off']
export_graphviz(DT17, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names=feature_names,class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
img= Image(graph.create_png())
graph.write_png("DT_10Feature_allData.png")

#-----------------------   Step 5: Logistic Regression  -----------------------
print('---------    Step 5: Logistic regression   ---------------------------')
# Training on Training data
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

preprocessing.normalize

LogReg=LogisticRegression()
LogReg.fit(X_train,Y_train['loan_status'])
LogReg.score(X_train,Y_train['loan_status'])

# 10 folds Cross-Validation on Training data
kf2 = KFold(len(X_train), n_folds=10)
scores2 = cross_val_score(LogReg, X_train, Y_train['loan_status'], scoring = 'accuracy', cv=kf2)
scores2.mean()
print('Logistic Regression cross-validation Accuracy:')
print(scores2)
print(scores2.mean())
'''
0.8356050303563937
'''

# Testing on Test Data
LogReg.score(X_test,Y_test)
print('Logistic Regression test set Accuracy:')
print(LogReg.score(X_test,Y_test))
'''
0.8327491236855283
'''
LogPre=LogReg.predict(X_test)
#sklearn.metrics.confusion_matrix(y_true, y_pred)
print('Logistic Regression on test dataset Confusion Matrix:')
cm = metrics.confusion_matrix(LogPre, Y_test,labels=[0,1])
print('            y_pred\n            0    1')
print('y_true 0   '+ '  ' . join(str(a) for a in cm[0]) + '  specificity:  %f'% (cm[0][0]/sum(cm[0])) )
print('       1   '+ '  ' . join(str(a) for a in cm[1]) + '  sensitivity:  %f'% (cm[1][1]/sum(cm[1])) )
'''
	        y_pred
            0    1
y_true 0   2361  436  specificity:  0.844119
       1   232  965  sensitivity:  0.806182
'''
#-------visualize Logistic Regression
#---: ROC Curve
y_pred_rt = LogReg.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(Y_test,y_pred_rt)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rt_lm, tpr_rt_lm, label='Logistic Regression',c='b')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#---: Scatter Plot
'''
for i in range(0,len(names)):
    name = names[i]
    scatter(X_test[:,i], y_pred_rt, s=4, c='deepskyblue')
    scatter(X_test[:,i], Y_test, marker='x', s=4, c='green')
    xlabel('scaled '+name)
    ylabel('Charge-off Rate')
    plt.title(name+' Charge-off Rate')
    legend(['Predicted Charge-off Rate','True Charge-off Rate'])
    plt.savefig('LogisticReg/Scaled'+str(i)+' '+name+' vs Label.png')

for i in range(0,len(names)):
    name = names[i]
    plt.scatter(X_train[:,i], LogReg.predict_proba(X_train)[:, 1], s=4)
    plt.xlabel(name)
    plt.ylabel('Predicted Charge-off Rate')
    plt.title('LogReg Model:'+' Predicted Charge-off Rate')
    legend(names)
    plt.savefig('LogisticReg/LogReg Model'+str(i)+' '+name+' vs Prob.png')
'''

# Scater plot on Training performance
if newPlots==1:
    for i in range(0,len(names)):
        name = names[i]
        plt.scatter(X_test[:,i], LogReg.predict_proba(X_test)[:, 1], s=4)
        plt.xlabel(name)
        plt.ylabel('Predicted Charge-off Rate')
        plt.title('LogReg Testing:'+' Predicted Charge-off Rate')
        legend(names)
        plt.savefig('LogisticReg/LogReg Testing'+str(i)+' '+name+' vs Prob.png')
        
    
    img=mpimg.imread('LogisticReg/LogReg Testing'+str(2)+' '+names[2]+' vs Prob.png')
    imgplot = plt.imshow(img)
    plt.show()
    img=mpimg.imread('LogisticReg/LogReg Testing'+str(5)+' '+names[5]+' vs Prob.png')
    imgplot = plt.imshow(img)
    plt.show()
    img=mpimg.imread('LogisticReg/LogReg Testing'+str(6)+' '+names[6]+' vs Prob.png')
    imgplot = plt.imshow(img)
    plt.show()

#-----------------------   Step 6: Clustering of Numerical Variable  ----------
print('---------     Step 6: Clustering of Numerical Variable ---------------')
data_new.reset_index(drop=True,inplace=True)

#divide data into numeric part and category part

df_num17=pd.DataFrame()#(13313,87)
df_catgr17=pd.DataFrame()#(13313,13)

for column in data_new.columns:
      if data_new[column]._is_numeric_mixed_type:
          df_num17[column]=data_new[column]
      else:
          df_catgr17[column]=data_new[column]

KM_data=df_num17

#scale the numermic data in [0,1]
scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(KM_data)

#divide the data into two clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_data)
pre=kmeans.predict(scaled_data)
pre=pd.DataFrame(pre)
pre['loan_status']=Status17Q1
pre.rename(columns={ pre.columns[0]: "cluster" }, inplace=True)

cluster1=pre[pre.cluster==0]
cluster2=pre[pre.cluster==1]

#Analysis in cluster
df_num17['cluster']=pre['cluster']
df_num17['sub_grade']=data_new['sub_grade']
cluster1_data=pd.DataFrame()
cluster1_data=df_num17[df_num17.cluster==0]
cluster2_data=pd.DataFrame()
cluster2_data=df_num17[df_num17.cluster==1]

cluster1_data.sub_grade.describe()
cluster2_data.sub_grade.describe()
#top sub_grade are both C1 in cluster1,2
cluster1_data.total_rec_prncp.describe()
'''
count     4062.000000
mean      8980.604764
std       8600.726921
min          0.000000
25%       3224.462500
50%       6062.335000
75%      11269.480000
max      40000.000000
'''
cluster2_data.total_rec_prncp.describe()
'''
count     9251.000000
mean      3305.107604
std       3065.413544
min          0.000000
25%       1309.710000
50%       2400.000000
75%       4082.960000
max      17000.000000
'''
cluster1_data.recoveries.describe()
'''
count     4062.000000
mean       516.736497
std       1575.612859
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max      39859.550000
Name: recoveries, dtype: float64
'''
cluster2_data.recoveries.describe()
'''
count     9251.000000
mean       184.197257
std        588.585269
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max      15758.650000
'''
cluster1_data.total_pymnt.describe()
'''
count     4062.000000
mean     12680.143718
std       8298.639335
min          0.000000
25%       7381.227500
50%      10115.220000
75%      15335.660000
max      50705.550050
Name: total_pymnt, dtype: float64
'''
cluster2_data.total_pymnt.describe()
'''
count     9251.000000
mean      4532.865598
std       3177.498127
min          0.000000
25%       2376.835000
50%       3895.796892
75%       5669.200000
max      18761.535620
Name: total_pymnt, dtype: float64
'''
cluster1_data.last_pymnt_amnt.describe()
'''
count     4062.000000
mean      4142.565238
std       8133.977999
min          0.000000
25%        658.497500
50%        832.470000
75%       1174.910000
max      40685.080000
Name: last_pymnt_amnt, dtype: float64
'''
cluster2_data.last_pymnt_amnt.describe()
'''
count     9251.000000
mean      1263.138187
std       2724.679596
min          0.000000
25%        219.880000
50%        335.690000
75%        493.860000
max      16813.450000
Name: last_pymnt_amnt, dtype: float64
'''



print("%.2f" %float((cluster1.loan_status==1).sum()/len(cluster1)*100)+"% of data in cluster1 is Charged Off")
print("%.2f" %float((cluster2.loan_status==1).sum()/len(cluster2)*100)+"% of data in cluster2 is Charged Off")

#Z0.05=1.645
p1=float((cluster1.loan_status==1).sum()/len(cluster1))
p2=float((cluster2.loan_status==1).sum()/len(cluster2))

##plot for cluster
pievalues_1=[p1,1-p1]
labels_1=['Charged Off','Not Charged Off']
colors_1=['blue', 'lightblue']
plt.pie(pievalues_1, labels=labels_1, colors=colors_1, startangle=90,explode=[0.1,0],autopct='%.2f%%')
plt.title('Charged Off Rate in cluster1')
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()

pievalues_2=[p2,1-p2]
labels_2=['Charged Off','Not Charged Off']
colors_2=['blue', 'lightblue']
plt.pie(pievalues_2, labels=labels_2, colors=colors_2, startangle=90,explode=[0.1,0],autopct='%.2f%%')
plt.title('Charged Off Rate in cluster2')
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()

n1=len(cluster1)
n2=len(cluster2)

Z=1.645
Min=(p1-p2)-Z*(np.sqrt(p1*(1-p1)/n1+p2*(1-p2)/n2))
Max=(p1-p2)+Z*(np.sqrt(p1*(1-p1)/n1+p2*(1-p2)/n2))

# alpha=0.1
#Min=-0.0132 Max=0.0161 -0.0132<p1-p2<0.0161, include 0


#-----------------   Step 6: KNN Classification of Numerical Variable  --------
#use the scaled numeric data to build KNN model; 
scaled_data=pd.DataFrame(scaled_data)

X_train_knn, X_test_knn, Y_train_knn, Y_test_knn = train_test_split(
        scaled_data, Status17Q1, test_size=0.3, random_state=42)

KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train_knn,Y_train_knn)
KNN_predicted=KNN.predict(X_train_knn)
KNN.score(X_train_knn,Y_train_knn)
'''
0.9472046356905247
'''
# Calculating 10 fold cross validation results on training data
kf3 = KFold(len(Y_train_knn),n_folds=10)
KNN_scores = cross_val_score(KNN, X_train_knn, Y_train_knn, cv=kf3)
'''
KNN_scores.mean()
0.902671915841105
'''
# test the model on test data
KNN.score(X_test_knn,Y_test_knn)
'''
0.8975963945918878
'''
KNNPre=KNN.predict(X_test_knn)
metrics.confusion_matrix(Y_test_knn, KNNPre, labels=[1,0])
'''
array([[1138,  263],
       [ 146, 2447]])

'''

#-----------------------   Step 7: Association of Categorical Variable  -------

# ---- Quantile-based discretization: convert numerical variable to bins
bins = 5 # Very High, High, Medium, Low, Very Low
for name in df_num17.columns:
    try:
        df_num17[name] = pd.qcut(df_num17[name], bins,
                labels=["VHH","H","M","L","VL"], duplicates='drop')
    # if the variable is no need to bins, skip it
    except Exception: 
        pass


# ---- Concat numerical and categirical variables together for Association
for name in df_catgr17.columns:
    df_num17[name] = df_catgr17[name]

# ---- add variable name for Association
    
for name in df_num17.columns:
    df_num17[name] = name +'='+ df_num17[name].astype(str)

asso = pd.DataFrame()
#Remove inrelevant and duplicated values
skip_num=['out_prncp_inv','recoveries']
for name in names:
    if name in skip_num:
        continue
    asso[name] = df_num17[name]

#Remove inrelevant and duplicated values
skip_cate=['disbursement_method','next_pymnt_d','title']
for name in df_catgr17.columns:
    if name in skip_cate:
        continue
    asso[name] = df_num17[name]

asso['loan_status'] = list(Y)
asso.to_csv('association variables.csv')

# ---- Association: apriori algorithm
pass
# please run apriori_loan_company.py with 'all variable.csv'

#-----------------------   Step 8: Visualization of the data  -----------------
###create 3D plot from common variable calculated by Univariate feature selection & Tree based selection
# total_rec_prncp,total_pymnt,last_pymnt_amnt,total_pymnt_inv,recoveries
ThreeD_list=['total_rec_prncp','last_pymnt_amnt','collection_recovery_fee']
ThreeD_com=list(itertools.combinations(ThreeD_list, 3))

i=0
while i<len(ThreeD_com):
    i=i+1
    ThreeD_P = pd.DataFrame()
    ThreeD_N = pd.DataFrame()
    ThreeD_list_current = ThreeD_com[i-1]
    j = 0
    while j < 3:
        j = j + 1
        ThreeD_P[ThreeD_list_current[j - 1]] = TrueP17Q1.loc[:, ThreeD_list_current[j - 1]]
        ThreeD_N[ThreeD_list_current[j - 1]] = TrueN17Q1.loc[:, ThreeD_list_current[j - 1]]

    ThreeD_P.reset_index(drop=True, inplace=True)
    ThreeD_N.reset_index(drop=True, inplace=True)
    ThreeD=plt.subplot(111,projection='3d')
    ThreeD.scatter(ThreeD_P[ThreeD_list_current[0]], ThreeD_P[ThreeD_list_current[1]], ThreeD_P[ThreeD_list_current[2]], c='r')
    ThreeD.scatter(ThreeD_N[ThreeD_list_current[0]], ThreeD_N[ThreeD_list_current[1]], ThreeD_N[ThreeD_list_current[2]], c='b')
    ThreeD.set_xlabel(ThreeD_list_current[0])
    ThreeD.set_ylabel(ThreeD_list_current[1])
    ThreeD.set_zlabel(ThreeD_list_current[2])
    ThreeD.set_title('pic num %i' % (i-1))
    ThreeD.legend(labels=['Charged Off','Not Charged Off'],loc='best')
    plt.show()

#we found that all Not Charged Off data's recoveries=0, half of Charged Off data's recoveries=0
#combine DT to tell in what situation, when recoveries=0, data=Charged Off

#for Not Charged off data, there is a linear relationship between total_pymnt and total_rec_prncp

