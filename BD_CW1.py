import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Scikit-learn library: For SVM
from sklearn import preprocessing
from sklearn.preprocessing import Imputer, scale
#from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


import itertools
sns.set(color_codes=True)

df = pd.read_csv("heart-attack/data.csv")
shapeold = df.shape
print (shapeold)


#display top 5 rows
print (df.head(5))

print df.dtypes


#identfy duplicates
duplicate_rows_df = df[df.duplicated()]
print "number of duplicate rows:", duplicate_rows_df.shape
print "Dupls:"
print duplicate_rows_df

# remove duplicaes
df = df.drop_duplicates()
print ("Duplicates removed")
print df.count()

# Replace all ? values with NaN
print('Replacing ? values with NaN...')
df.replace('?', np.NaN, inplace=True)

df.replace(' ?', np.NaN, inplace=True)

print(df.dtypes)
# Convert to correct data types
df = df.astype({"trestbps": float, "chol": float, "fbs": float, "slope": float,
                "restecg": float, "thalach": float, "exang": float})
print(df.dtypes)

# Count number of null values
print("Null values:")
print(df.isnull().sum())


# column ca, thal should be removed
df.drop(['ca', 'slope', 'thal'], axis=1, inplace=True) 
print("Null values:")
print(df.isnull().sum())
print(df.shape)



# Handle missing values 
# Replace missing values in chol and thalach with column mean
df['chol'] = df['chol'].fillna(df['chol'].mean())
df['thalach'] = df['thalach'].fillna(df['thalach'].mean())

# Remove null from fbs restecg, exang
df = df.dropna(subset=['fbs', 'exang', 'restecg'])
#
#print('Null Values removed')
print(df.isnull().sum())
print df.shape

#
## Test each column for any unique values that should not be present
#print('Unique values in column')
#print(df['age'].unique())
#print(df['sex'].unique())
#print(df['cp'].unique())
#print(df['trestbps'].unique())
#print(df['chol'].unique())
#print(df['fbs'].unique())
#print(df['restecg'].unique())
#print(df['thalach'].unique())
#print(df['exang'].unique())
#print(df['oldpeak'].unique())
#print(df['target'].unique())


#ADD k means clustering


def find_outliers_tukey(x):
    q1 = np.percentile(x,25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1
    floor  =q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])
    
    return outlier_indices, outlier_values
    
#
#def find_outliers_kde(x):
#    x_scaled = scale(list(map(float, x)))
#    kde = KDEUnivariate(x_scaled)
#    kde.fit(bw="scott", fft=True)
#    pred = kde.evaluate(x_scaled)
#    
#    n = sum(pred < 0.05)
#    outlier_ind = np.asarray(pred).argsort()[:n]
#    outlier_value = np.asarray(x)[outlier_ind]
#    
#    return outlier_ind, outlier_value
#    
#    
pos = ((df['target'] == 1).sum())
neg = ((df['target'] == 0).sum())
rate = float(pos) / (float(neg) + float(pos))

print ('Diagnosed heart disease: ')
print pos
print ('Diagnosed no heart disease:')
print neg
print ('rate:')
print rate
print('\n')

#sns.boxplot(x=df['chol'])

print (" ")
print ("chol tukey outliers:")
tukey_indices, tukey_values = find_outliers_tukey(df['chol'])
print (np.sort(tukey_values))

print(df.shape)
print ('Removing outliers \n')
df = df[df.chol < 600]
print(df.shape)


corl = df.corr()
plt.figure(figsize=(15,10))
#sns.heatmap(corl, cmap="YlGnBu", annot=True)
print 'corl:'
print corl

rank = corl['target']
df_rank = pd.DataFrame(rank)
df_rank = np.abs(df_rank).sort_values(by='target', ascending=False)
print "df rank:"
print df_rank
print('\n')

print df.shape
df_train_all = df[0:283]

#Training 
print df_train_all

df_train_1 = df_train_all[df_train_all['target'] == 1]
df_train_0 = df_train_all[df_train_all['target'] == 0]

print('In this dataset we have ' + str(len(df_train_1)) +" at risk of heart disease")
print('In this dataset we have ' + str(len(df_train_0)) +" not at risk of heart disease")
df_sample = df_train_0.sample(85)
df_train = df_train_1.append(df_sample)
df_train = df_train.sample(frac=1)
xtrain = df_train.drop(['target'], axis=1)
ytrain = df_train['target']
xtrain = np.asarray(xtrain)
ytrain = np.asarray(ytrain)
print(xtrain)

df_test_all = df[86:]
#print ('check df test all')
#print df_test_all
x_test_all = df_test_all.drop(['target'], axis=1)
y_test_all = df_test_all['target']
x_test_all = np.asarray(x_test_all)
y_test_all = np.asarray(y_test_all)

x_train_rank = df_train[df_rank.index[1:10]]
x_train_rank = np.asarray(x_train_rank)
x_test_all_rank = df_test_all[df_rank.index[1:10]]
x_test_all_rank = np.asarray(x_test_all_rank)
y_test_all = np.asarray(y_test_all)



class_names = np.array(['0','1'])

def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes)) 
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('predicted label')
        
#classifier = svm.SVC(kernel= 'rbf', gamma=1000, decision_function_shape='ovr')
classifier = svm.SVC(kernel= 'linear', degree = 1, gamma=100, decision_function_shape='ovo', C=1000, class_weight=None, tol=0.001)
classifier.fit(xtrain, ytrain)
print (classifier.fit) 
prediction_SVM_all = classifier.predict(x_test_all)
print ()
cm = confusion_matrix(y_test_all, prediction_SVM_all)
plot_confusion_matrix(cm, class_names)

classifierscore = classifier.score(x_test_all, y_test_all)
print("classifier score")
print classifierscore


#print (sns.pairplot(df, hue='y', kind="reg"))