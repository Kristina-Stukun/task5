import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 

pd.set_option("display.max_columns", 8)
data= pd.read_csv('data.csv')
df  = data.copy()
##########Сократить датасет##########
qw = df.copy()
#print(qw.isnull().sum(axis=1)/15*100)
qw['процент']=df.isnull().sum(axis=1)/15*100
#print(qw)

df = qw.loc[qw['процент']<70]
#print(df)

############Часть 2##################
le = LabelEncoder()
counts = 0
encod = []
f = open('group.txt', 'r', encoding = 'utf-8')
data = f.readlines()
for line in data:
    line=line.replace('\n','')
    counts+=1
    if  counts % 2 != 0:
        encod.append(line)
f.close()        


f = open('encoder.txt', 'w', encoding = 'utf-8')
for i in le.fit_transform(encod):
    f.write(str(i) )
    f.write('\n')   
f.close()
WEB= ['web','веб','full stack','fullstack','front-end','frontend','backend','back-end','верстальщик']
TEST = ['тестировщик','qa','тестированию','тестирование']
MARK= ['маркетолог','таргетолог']
SUP=['техподдержки','службы поддержки']
PROG= ['программист','разработчик','developer','programmer']
SIS = ['системный администратор']
A= ['аналитик','анализ']
ENJ=['инженер','engineer']
SPEC = ['специалист','specialist']
MAN=['менеджер']

df['класс']=0
clases = []
nam =df['name'].copy()
for i in nam:
    if len([True for x in WEB if x in i.lower()])!=0:
        clases.append(0)
    elif len([True for x in TEST if x in i.lower()])!=0:
        clases.append(10)
    elif len([True for x in MARK if x in i.lower()])!=0:
        clases.append(4)
    elif len([True for x in SUP if x in i.lower()])!=0:
        clases.append(9)
    elif len([True for x in PROG if x in i.lower()])!=0:
        clases.append(6)
    elif len([True for x in SIS if x in i.lower()])!=0:
        clases.append(7)
    elif len([True for x in A if x in i.lower()])!=0:
        clases.append(1)
    elif len([True for x in ENJ if x in i.lower()])!=0:
        clases.append(2)
    elif len([True for x in SPEC if x in i.lower()])!=0:
        clases.append(8)
    elif len([True for x in MAN if x in i.lower()])!=0:
        clases.append(5)
    else:
        clases.append(3)
se = pd.Series(clases)
df['класс'] = se.values

###########предобработка###############
###########удаляем незначащие признаки, кодируем категориальные, заполняем пропуски,нормализуем###################
df.drop(['address','company', 'date','description','url','процент','duty','conditions','requirements','skills','name'], axis='columns', inplace=True)
df.type_of_employment=le.fit_transform(df.type_of_employment)
df.experience=le.fit_transform(df.experience)
df.schedule=le.fit_transform(df.schedule)
df.salaryMAX = df.salaryMAX.fillna(round(df.salaryMAX.mean(),1))
df.salaryMIN = df.salaryMIN.fillna(round(df.salaryMIN.mean(),1))
#df=(df-df.min())/(df.max()-df.min())
#df = df.reset_index()
##########разделение на тестовую и обучающую#############
X= df.iloc[:, :-1].values
y = df.iloc[:, 5].values
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=0)
#####выбрать минимум 3 классификатора#####
########k-соседей, метод опорных векторов, дерево решений ##########################

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

SVC_model = SVC()
DTC_model = DecisionTreeClassifier() 
KNN_model = KNeighborsClassifier(n_neighbors=5)  

SVC_model.fit(X_train, y_train) 
KNN_model.fit(X_train, y_train)
DTC_model.fit(X_train, y_train)

SVC_prediction = SVC_model.predict(X_test) 
KNN_prediction = KNN_model.predict(X_test)
DTC_prediction = DTC_model.predict(X_test)
# Оценка точности — простейший вариант оценки работы классификатора
print(accuracy_score(SVC_prediction, y_test))  
print(accuracy_score(KNN_prediction, y_test))  
print(accuracy_score(DTC_prediction, y_test))  
############Часть 2.2 ##############################################
data2= pd.read_csv('6.csv')
df2  = data2.copy()
df2['класс']=0
clases = []
nam =df2['name'].copy()
for i in nam:
    if len([True for x in WEB if x in i.lower()])!=0:
        clases.append(0)
    elif len([True for x in TEST if x in i.lower()])!=0:
        clases.append(10)
    elif len([True for x in MARK if x in i.lower()])!=0:
        clases.append(4)
    elif len([True for x in SUP if x in i.lower()])!=0:
        clases.append(9)
    elif len([True for x in PROG if x in i.lower()])!=0:
        clases.append(6)
    elif len([True for x in SIS if x in i.lower()])!=0:
        clases.append(7)
    elif len([True for x in A if x in i.lower()])!=0:
        clases.append(1)
    elif len([True for x in ENJ if x in i.lower()])!=0:
        clases.append(2)
    elif len([True for x in SPEC if x in i.lower()])!=0:
        clases.append(8)
    elif len([True for x in MAN if x in i.lower()])!=0:
        clases.append(5)
    else:
        clases.append(3)
se = pd.Series(clases)
df2['класс'] = se.values
df3 = df2.copy()
df2.drop(['address','company', 'date','description','url','duty','conditions','requirements','skills','name'], axis='columns', inplace=True)
df2.type_of_employment=le.fit_transform(df2.type_of_employment)
df2.experience=le.fit_transform(df2.experience)
df2.schedule=le.fit_transform(df2.schedule)
df2.salaryMAX = df2.salaryMAX.fillna(round(df2.salaryMAX.mean(),1))
df2.salaryMIN = df2.salaryMIN.fillna(round(df2.salaryMIN.mean(),1))
df2=(df2-df2.min())/(df2.max()-df2.min())

X_train= df.iloc[:, :-1].values
y_train = df.iloc[:, 5].values
X_test= df2.iloc[:, :-1].values
y_test= df2.iloc[:, 5].values
DTC_model = DecisionTreeClassifier() 
DTC_model.fit(X_train, y_train)
DTC_prediction = DTC_model.predict(X_test)
df3['предпологаемая']=DTC_prediction
df3[['name', 'класс', 'предпологаемая']].to_csv('itog.csv')