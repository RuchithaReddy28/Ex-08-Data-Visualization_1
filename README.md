# Ex-08-Data-Visualization-

## AIM
To Perform Data Visualization on a complex dataset and save the data to a file. 

# Explanation
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature generation and selection techniques to all the features of the data set
### STEP 4
Apply data visualization techniques to identify the patterns of the data.


# CODE
```
Name:Akkireddy Ruchitha Redddy
Register No:212221230004
```
```
#loading the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Superstore.csv")
df

#removing unnecessary data variables
df.drop('Row ID',axis=1,inplace=True)
df.drop('Order ID',axis=1,inplace=True)
df.drop('Customer ID',axis=1,inplace=True)
df.drop('Customer Name',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)
df.drop('Postal Code',axis=1,inplace=True)
df.drop('Product ID',axis=1,inplace=True)
df.drop('Product Name',axis=1,inplace=True)
df.drop('Order Date',axis=1,inplace=True)
df.drop('Ship Date',axis=1,inplace=True)
print("Updated dataset")
df

df.isnull().sum()

#detecting and removing outliers in current numeric data
plt.figure(figsize=(12,10))
plt.title("Data with outliers")
df.boxplot()
plt.show()

plt.figure(figsize=(12,10))
cols = ['Sales','Quantity','Discount','Profit']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

#data visualization
#line plots
import seaborn as sns
sns.lineplot(x="Sub-Category",y="Sales",data=df,marker='o')
plt.title("Sub Categories vs Sales")
plt.xticks(rotation = 90)
plt.show()

sns.lineplot(x="Category",y="Profit",data=df,marker='o')
plt.xticks(rotation = 90)
plt.title("Categories vs Profit")
plt.show()

sns.lineplot(x="Region",y="Sales",data=df,marker='o')
plt.xticks(rotation = 90)
plt.title("Region area vs Sales")
plt.show()

sns.lineplot(x="Category",y="Discount",data=df,marker='o')
plt.title("Categories vs Discount")
plt.show()

sns.lineplot(x="Sub-Category",y="Quantity",data=df,marker='o')
plt.xticks(rotation = 90)
plt.title("Sub Categories vs Quantity")
plt.show()

#bar plots
sns.barplot(x="Sub-Category",y="Sales",data=df)
plt.title("Sub Categories vs Sales")
plt.xticks(rotation = 90)
plt.show()

sns.barplot(x="Category",y="Profit",data=df)
plt.title("Categories vs Profit")
plt.show()

sns.barplot(x="Sub-Category",y="Quantity",data=df)
plt.title("Sub Categories vs Quantity")
plt.xticks(rotation = 90)
plt.show()

sns.barplot(x="Category",y="Discount",data=df)
plt.title("Categories vs Discount")
plt.show()

plt.figure(figsize=(12,7))
sns.barplot(x="State",y="Sales",data=df)
plt.title("States vs Sales")
plt.xticks(rotation = 90)
plt.show()

plt.figure(figsize=(25,8))
sns.barplot(x="State",y="Sales",hue="Region",data=df)
plt.title("State vs Sales based on Region")
plt.xticks(rotation = 90)
plt.show()

#Histogram
sns.histplot(data = df,x = 'Region',hue='Ship Mode')
sns.histplot(data = df,x = 'Category',hue='Quantity')
sns.histplot(data = df,x = 'Sub-Category',hue='Category')
plt.xticks(rotation = 90)
plt.show()
sns.histplot(data = df,x = 'Quantity',hue='Segment')
plt.hist(data = df,x = 'Profit')
plt.show()

#count plot
plt.figure(figsize=(10,7))
sns.countplot(x ='Segment', data = df,hue = 'Sub-Category')
sns.countplot(x ='Region', data = df,hue = 'Segment')
sns.countplot(x ='Category', data = df,hue='Discount')
sns.countplot(x ='Ship Mode', data = df,hue = 'Quantity')

#Barplot 
sns.boxplot(x="Sub-Category",y="Discount",data=df)
plt.xticks(rotation = 90)
plt.show()
sns.boxplot( x="Profit", y="Category",data=df)
plt.xticks(rotation = 90)
plt.show()
plt.figure(figsize=(10,7))
sns.boxplot(x="Sub-Category",y="Sales",data=df)
plt.xticks(rotation = 90)
plt.show()
sns.boxplot(x="Category",y="Profit",data=df)
sns.boxplot(x="Region",y="Sales",data=df)
plt.figure(figsize=(10,7))
sns.boxplot(x="Sub-Category",y="Quantity",data=df)
plt.xticks(rotation = 90)
plt.show()
sns.boxplot(x="Category",y="Discount",data=df)
plt.figure(figsize=(15,7))
sns.boxplot(x="State",y="Sales",data=df)
plt.xticks(rotation = 90)
plt.show()

#KDE plot
sns.kdeplot(x="Profit", data = df,hue='Category')
sns.kdeplot(x="Sales", data = df,hue='Region')
sns.kdeplot(x="Quantity", data = df,hue='Segment')
sns.kdeplot(x="Discount", data = df,hue='Segment')

#violin plot
sns.violinplot(x="Profit",data=df)
sns.violinplot(x="Discount",y="Ship Mode",data=df)
sns.violinplot(x="Quantity",y="Ship Mode",data=df)

#point plot
sns.pointplot(x=df["Quantity"],y=df["Discount"])
sns.pointplot(x=df["Quantity"],y=df["Category"])
sns.pointplot(x=df["Sales"],y=df["Sub-Category"])

#Pie Chart
df.groupby(['Category']).sum().plot(kind='pie', y='Discount',figsize=(6,10),pctdistance=1.7,labeldistance=1.2)
df.groupby(['Sub-Category']).sum().plot(kind='pie', y='Sales',figsize=(10,10),pctdistance=1.7,labeldistance=1.2)
df.groupby(['Region']).sum().plot(kind='pie', y='Profit',figsize=(6,9),pctdistance=1.7,labeldistance=1.2)
df.groupby(['Ship Mode']).sum().plot(kind='pie', y='Quantity',figsize=(8,11),pctdistance=1.7,labeldistance=1.2)

df1=df.groupby(by=["Category"]).sum()
labels=[]
for i in df1.index:
    labels.append(i)  
plt.figure(figsize=(8,8))
colors = sns.color_palette('pastel')
plt.pie(df1["Profit"],colors = colors,labels=labels, autopct = '%0.0f%%')
plt.show()

df1=df.groupby(by=["Ship Mode"]).sum()
labels=[]
for i in df1.index:
    labels.append(i)
colors=sns.color_palette("bright")
plt.pie(df1["Sales"],labels=labels,autopct="%0.0f%%")
plt.show()

#HeatMap
df4=df.copy()

#encoding
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder
oe=OrdinalEncoder()

df4["Ship Mode"]=oe.fit_transform(df[["Ship Mode"]])
df4["Segment"]=oe.fit_transform(df[["Segment"]])
df4["City"]=le.fit_transform(df[["City"]])
df4["State"]=le.fit_transform(df[["State"]])
df4['Region'] = oe.fit_transform(df[['Region']])
df4["Category"]=oe.fit_transform(df[["Category"]])
df4["Sub-Category"]=le.fit_transform(df[["Sub-Category"]])

#scaling
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df5=pd.DataFrame(sc.fit_transform(df4),columns=['Ship Mode', 'Segment', 'City', 'State','Region',
                                               'Category','Sub-Category','Sales','Quantity','Discount','Profit'])

#Heatmap
plt.subplots(figsize=(12,7))
sns.heatmap(df5.corr(),cmap="PuBu",annot=True)
plt.show()
```

# OUPUT
##mInitial Dataset:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds1.png?raw=true)
## Cleaned Dataset:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds2.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds3.png?raw=true)
## Removing Outliers:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds4.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds5.png?raw=true)
## Line PLot:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds6.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds7.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds8.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds9.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds10.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds11.png?raw=true)
## Bar Plots:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds12.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds13.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds14.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds15.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds16.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds17.png?raw=true)
## Histograms:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds18.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds19.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds20.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds21.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds22.png?raw=true)
## Count plots:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds23.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds24.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds25.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds26.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds27.png?raw=true)
## Bar Charts:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds28.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds29.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds30.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds31.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds32.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds33.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds34.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds35.png?raw=true)
## KDE Plots:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds36.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds37.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds38.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds39.png?raw=true)
## Violin Plot:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds40.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds41.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds42.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds43.png?raw=true)
## Point Plots:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds44.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds45.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds46.png?raw=true)
## Pie Charts:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds47.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds48.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds49.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds50.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds51.png?raw=true)
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds52.png?raw=true)
## HeatMap:
![output](https://github.com/RuchithaReddy28/Ex-08-Data-Visualization_1/blob/main/ds53.png?raw=true)
# Result:
Hence,Data Visualization is applied on the complex dataset using libraries like Seaborn and Matplotlib successfully and the data is saved to file.
