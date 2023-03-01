#!/usr/bin/env python
# coding: utf-8

# ![clustering.jpg](attachment:clustering.jpg)
# 
# # FASHION DATA ANALYTICS; Market Segmentation with KMeans Clustering 
# <h31><center>by</center></h3>
# <h1><center>Olaoluwakiitan Olabiyi</center></h1>

# # SESSION 1: Data Importation and Prepartion¶ 
# 
# # Import the libraries

# In[82]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype,is_numeric_dtype

import category_encoders as ce
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
pd.set_option('display.max_columns', None)


# # Load the data

# In[2]:


# Load the data
fda_df = pd.read_csv ('retail_dataset.csv', encoding="ISO-8859-1") #I added the encosing because I got a unicode error while loading the data.


# In[3]:


fda_df.head()              


# In[4]:


# Check the number of rows and columns ; 31 columns, 3675 rows
fda_df.shape


# In[5]:


# Quick information about the datatypes and number of rows
fda_df.info() # the datatypes present are ; object, float and int


# # Data Preprocessing

# In[6]:


# make a copy of the data
fda= fda_df.copy()

# drop irrelevant columns
fda = fda.dropna(subset= ['Order ID'])
            
fda.isnull().sum() # to recheck nulls


# ### Notes for missing values
# 
# The following variables contain 2 missing values
# - Purchase Frequency
# - Order ID    
# - AOV
# 
# I will drop the rows with nulls in the order ID and recheck if there are still missing values.

# In[7]:


fda.isnull().sum() #check for missing values


# ### I will drop the following variables because thaey are either duplicates or not related to my analysis.
# 
# "Consumer ID","Consumer ID (Purchases info)","Month",
#            'Year','%Customer','%GP','Gross_Profit',
#            'Net_Sales','Line Value (net discount)','Period'

# In[8]:


# Drop the irrelevant variables;
fda = fda.drop(["Consumer ID","Consumer ID (Purchases info)","Month",
           'Year','%Customer','%GP','Gross_Profit',
           'Net_Sales','Line Value (net discount)','Period'], axis=1)


# ### I noticed some variables were not assigned the right data type, hence , I need to change their data type to the right one.

# In[9]:


fda['New active'] = fda['New active'].astype(str)
fda['New repeaters'] = fda['New repeaters'].astype(str)
fda['Marketing optin '] = fda['Marketing optin'].astype(str)
fda['Year (Purchases info)'] = fda['Year (Purchases info)'].astype(str)
fda['Month (Purchases info)'] = fda['Month (Purchases info)'].astype(str)


# In[10]:


fda.head()


# In[11]:


# Check the shape of the new dataframe

fda.shape


# ### After the data preparation,  it would be helpful to see some descriptive statistics to understand the patterns in the data.

# # SESSION 2:Basic Statistics
# 
# ### I will start will the numeric variables and then explore the categorical features.

# In[12]:


# Divide the categorical and numeric variables
num_list= []
cat_list= []

for feat in fda:
    if is_numeric_dtype(fda[feat]):
        num_list.append(feat)
    elif is_string_dtype(fda[feat]):
        cat_list.append(feat)
        
print("numeric: ", num_list)
print("Categorical: ",cat_list)


# In[13]:


num_list


# In[14]:


fda.describe()


# #### The minimum ,maximum and averge order per customer is 1,10 and 2 respectively. I addition, ordrvalue has a minimum, maximum and average of £144, £1576 and £388 respectively.

# In[15]:


cat_list ### I am interested in the gender, age range, line category and country


# In[16]:


print (fda['Age Range'].value_counts())
print (round(fda['Age Range'].value_counts(normalize=True) * 100),2) ## result in percentage
print (" ")

print (fda['Gender'].value_counts())
print (round(fda['Gender'].value_counts(normalize=True) * 100),2) ## result in percentage
print (" ")

print (fda['Line Category'].value_counts())
print (round(fda['Line Category'].value_counts(normalize=True) * 100),2) ## result in percentage
print (" ")

print (fda['Country'].value_counts())
print (round(fda['Country'].value_counts(normalize=True) * 100),2) ## result in percentage
print (" ")


# ## Insights;
# 
# - 42%(1560) of the customers are betwen the age of 18-24, while 40+ represent only 7% (245)
# - 93% (3399) of K&C customers are females
# - With respect to the product category, Fragrances(731),Apparel(709),Small Leather goods(537),Accessories(514),Sneakers(316) are the top 5 popularly sold producst with a percentage of 20%,19%,15% and 14% respectively.
# - Lastly, 82% of the purchases are from Italy.(NB: The business was only in Italy in 22019, but opened new branches in other countries in 2020)

# # SESSION 3: Exploratory Data Analysis

# ### Next, I want to plot some graphs to spot more trends. I will start with univariate, bivariate and lastly multivariate analysis.

# # Univariate Analysis

# ## What age group do we have the most number of customers?

# In[17]:


plt.figure (figsize= (10, 6))
sns.set_style('whitegrid')
sns.set(font_scale= 1.5)
aplot=sns.countplot(fda['Age Range'], label = 'counts',order=fda['Age Range'].value_counts().index, palette= 'viridis')  

# label the plot
for p in aplot.patches:
    aplot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title ('Customer Age Distribution') 


# ###  ~ 42% of the customers are within the 18-24 age group

# ## How are our customers distributed by country?

# In[18]:


plt.figure (figsize= (10, 6))
sns.set_style('whitegrid')
sns.set(font_scale= 1.5)
cplot=sns.countplot(fda['Country'], label = 'counts',order=fda['Country'].value_counts().index, palette= 'dark')  #2. StartYear
plt.xticks(rotation=45)

# label the plot
for p in cplot.patches:
    cplot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title ('Customer Distribution by Country') 


# ### 82% of the customer purchases come from Italy

# ## What is our top three products by purchase?

# In[19]:


plt.figure (figsize= (10, 8))
sns.set_style('whitegrid')
sns.set(font_scale= 1.5)
pplot= sns.countplot(fda['Line Category'], label = 'counts',order=fda['Line Category'].value_counts().index, palette= 'bright')  #2. StartYear
plt.xticks(rotation=45)

# label the plot
for p in pplot.patches:
    pplot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title ('Product Categories') 


# ### The top three products by purchases are Fragrance, Apparel and Leather Goods.

# ## What is the gender distribution of our customers?

# In[20]:


plt.figure (figsize= (14, 6))
sns.set_style('whitegrid')
sns.set(font_scale= 1.5)
gplot=sns.countplot(fda['Gender'], label = 'counts',order=fda['Gender'].value_counts().index, palette= 'Set2') 

# label the plot
for p in gplot.patches:
    gplot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title ('Customer Distribution by Gender') 


# ### ~93% of the customers are females.

# ### Next, I will explore some of the numeric variables

# - I am interested in the AOV,Consumer LTV, Discount, Net Sales, Gross Profit, Line SKU Production and the Total Order Value.

# In[21]:


plt.figure(figsize= [24,20]) # set the subplots  size

plt.subplot(2,3,1)
sns.histplot(data=fda, x="Net Sales", bins=20)
plt.title('Net Sales Chart')
plt.xlabel('Net Sales(£)')
plt.ylabel('Count')

plt.subplot(2,3,2)
sns.histplot(data=fda, x="Gross Profit", bins= 20)
plt.title('Gross Profit Chart')
plt.xlabel('Gross Profit(£)')
plt.ylabel('Count')

plt.subplot(2,3,3)
sns.histplot(data=fda, x="Discount", bins= 20)
plt.title('Discount Chart')
plt.xlabel('Discount(£)')
plt.ylabel('Count')

plt.subplot(2,3,4)
sns.histplot(data=fda, x="AOV", bins= 20)
plt.title('Average Order Value Chart')
plt.xlabel('AOV(£)')
plt.ylabel('Count')

plt.subplot(2,3,5)
sns.histplot(data=fda, x="Line SKU Production Cost", bins= 20)
plt.title('Line SKU Production Cost Chart')
plt.xlabel('Line SKU Production Cost(£)')
plt.ylabel('Count')

plt.subplot(2,3,6)
sns.histplot(data=fda, x="Total order value", bins= 20)
plt.title('Total Order Value Chart')
plt.xlabel('Total order value(£)')
plt.ylabel('Count')


# ### Insights;
# 
# - The histogram of the values presented above are right skewed and also display a form of bimodal distribution. 
# 
# - The Net Sales([Total order value]- [Discount]) shows that there are more sales with a lesser net sale value.In addition, the plot rrveals a different group of data from £700 to about £1500.
# 
# - A similar pattern was observed in the gross profit.1000 sales generated nearly £200 and less than 20 generated about £1000.
# 
# - For the discount, over 2000 sales had little or no discount. The highest discount is £315 and less than 10 sales had such discounst.
# 
# - The average order value also shows a bimodal distribution with about 1000 sales having roughly £100-£250 AOV.
# 
# - Similarly, there are more sales with a total order value less than £500. 
# 
# - There are also more sales with a lower production cost.
# 

# # Bivariate Analysis
# 
# #### In this session, I will compare numerical variables with respect to gender  categories.

# In[22]:


rnd_pro= round(fda.groupby('Gender')['Line Category'].value_counts(normalize=True) *100,0)

print(pd.DataFrame(fda.groupby('Gender')['Line Category'].value_counts()))
print(" ")
print(pd.DataFrame(rnd_pro))
print(" ")


rnd_age= round(fda.groupby('Gender')['Age Range'].value_counts(normalize=True) *100,0)

print(pd.DataFrame(fda.groupby('Gender')['Age Range'].value_counts()))
print(" ")
print(pd.DataFrame(rnd_age))
print(" ")

rnd_cnt= round(fda.groupby('Gender')['Country'].value_counts(normalize=True) *100,0)

print(pd.DataFrame(fda.groupby('Gender')['Country'].value_counts()))
print(" ")
print(pd.DataFrame(rnd_cnt))
print(" ")


# #### Insights;
# 
# - The top three popular products in female fragrances(20%),apparel(19%), small leather goods(15%), while in the males often buy apparel(23%), sandals(18%) and fragrances(15%). 
# 
# - In otherwords,the sales of apparel and sandals are higher in the male gender while the femalse purchase more have a higher proprtion of fragrance sold.
# 
# - 42% of each gender category are within 18-24 age range.
# 
# - Despite the fact that 93% of the customers are male, the age range distribution is fairly equal in both gender categories.
# 
# - 87% of the female customers are from Italy, while the highest number of male customers(20%) are from Saudi Arabia.

#  #### It would be helpful to visualize the relationship between the numeric variables with respect to the gender categories. Following this, I will compare the relationhip between the categorical variables.

# In[23]:


plt.figure(figsize=[24, 20])

plt.subplot(2,3,1)
sns.boxplot(data=fda, x='Gender', y='Net Sales', palette= 'Set2', linewidth=2)
plt.xticks(rotation=0);
plt.xlabel('Gender')
plt.ylabel('Net Sales(£)')
plt.title('Net Sales by Gender')

plt.subplot(2,3,2)
sns.boxplot(data=fda, x='Gender', y='Gross Profit', palette= 'Set2', linewidth=2)
plt.xticks(rotation=0);
plt.xlabel('Gender')
plt.ylabel('Gross Profit(£)')
plt.title('Gross Profit by Gender')

plt.subplot(2,3,3)
sns.boxplot(data=fda, x='Gender', y='Discount', palette= 'Set2', linewidth=2)
plt.xticks(rotation=0);
plt.xlabel('Gender')
plt.ylabel('Discount(£)')
plt.title('Discount by Gender')

plt.subplot(2,3,4)
sns.boxplot(data=fda, x='Gender', y='Total order value', palette= 'Set2', linewidth=2)
plt.xticks(rotation=0);
plt.xlabel('Gender')
plt.ylabel('Total order value(£)')
plt.title('Total Order Value by Gender')

plt.subplot(2,3,5)
sns.boxplot(data=fda, x='Gender', y='AOV', palette= 'Set2', linewidth=2)
plt.xticks(rotation=0);
plt.xlabel('Gender')
plt.ylabel('Average order value(£)')
plt.title('Average Order Value by Gender')

plt.subplot(2,3,6)
sns.boxplot(data=fda, x='Gender', y='Line SKU Production Cost', palette= 'Set2', linewidth=2)
plt.xticks(rotation=0);
plt.xlabel('Gender')
plt.ylabel('Line SKU Production Cost(£)')
plt.title('Line SKU Production Cost')


# ### Insights;
# 
# #### Right off the bat, the outliers and skewness of the data can be spotted , however, I would still interprete the patters in the plots above.
# 
# - Though the females have a higher net sales, the males median net sales were higher.
# 
# - Despite the fact that there are significantly more female customers, the highest and median gross profit for both gender categories is fairly the same. Could it be that the males buy of the products with the highest gross profits?
# 
# - With resspect to the discount, the median line for the female is absent. This is because the discount variable is skewed, hence the median line for the female aligns with either of the quartile values.
# 
# - The females have a lower discount range compared to the males.
# 
# - For the total and average order values, the males have a higher median and slightly higher value.
# 
# - The range for the Line SKU Production is fairly similar in both genders. Although, with a higher median value in the male.
# 

# #### I have plotted a few of the categorical variables to further illustrate some of the relationships I explained earlier.
# 

# In[24]:


plt.figure (figsize= (10,6), tight_layout= True)
plt.title ('Product Sales by Gender')
sns.countplot(y= 'Line Category', hue= 'Gender', data =fda, palette= 'Set2');
plt.show()


# In[25]:


plt.figure (figsize= (10,8), tight_layout= True)
plt.title ('Product Sales by Age Group')
sns.countplot (y= 'Line Category', hue= 'Age Range', data =fda, palette= 'RdPu');
plt.show()


# ## Multivariate Analysis

# In[26]:


correlation= fda[num_list].corr()# correlation plot
plt.figure(figsize = [12, 8])
sns.heatmap(correlation, annot=True, fmt = '.3f',cmap = 'Set3', center = 0)


# ### A correlation plot is an interesting way to view what is happening between the numeric values in one image.
# 
# From the heatmap above,there is a very small correlation between the customer lifetime value and the discount. However, the gross profit has a medium size correlation.
# 
# Statistical analysis, Pearson Correlation Test, would be able to explain this better, with the p-values.

# In[27]:


plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data= fda, x='Line Category', y='Net Sales', alpha= 0.6 , hue='Gender', palette='Set2', s=70)
ax.set(xlabel='Product Category', ylabel='Net Sales')
plt.title('Net Sales by Product Categories')
plt.xticks(rotation=45)


# In[28]:


plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data= fda, x='Line Category', y='Gross Profit', alpha= 0.6 , hue='Gender', palette='Set2', s=70)
ax.set(xlabel='Product Category', ylabel='Gross Profit')
plt.title('Gross Profit by Product Category')
plt.xticks(rotation=45)


# In[29]:


plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data= fda, x='Line Category', y='Discount', alpha= 0.6 , hue='Gender', palette='Set2', s=70)
ax.set(xlabel='Product Category', ylabel='Discount')
plt.title('Discount by Product Category')
plt.xticks(rotation=45)


# #### The net sales, gross sales and discount were higher for mini bags and bags sold among both genders. Fragrances and Small leather goods generated the least net sales, even though those were the most popular products in the female customers.
#  - More insights would have been revealed if the dataset had included the product prices.

# In[30]:


plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data= fda, x='Age Range', y='Gross Profit', alpha= 0.6 , hue='Gender', palette='Set2', s=70)
ax.set(xlabel='Age Group', ylabel='Gross Profit')
plt.title('Gross Profit by Customer Age Group')
plt.xticks(rotation=45)


# #### Despite the fact that 42% of the customers are within the 18-24 age range, the 25-29 age range, which is only 17% of the customers,generate the highest gross profit.
# 
# Could it be that the 18-24 age group buy more cheaper products? or products with produced in high quantity but less profit?

# In[31]:


plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data= fda, x='Age Range', y='Net Sales', alpha= 0.6 , hue='Gender', palette='Set2', s=70)
ax.set(xlabel='Age Group', ylabel='Gross Profit')
plt.title('Net Sales by Customer Age Group')
plt.xticks(rotation=45)


# ### Note:
# 
# A pair plot is a great way to quickly visualize the relationships between the numeric variables, with respect to a categorical. variable. It helps me to easily spot correlations, similar to a heatmap.

# In[32]:


sns.set_style('ticks')
sns.pairplot(fda,vars=['Gross Profit','Net Sales','Discount','Line SKU Production Cost'], hue='Gender', diag_kind='kde', kind='scatter', palette='Set2')
plt.show()


# ### A similar correlation trend is observed in both the male and female categories and similar to what was observed in the heatmap.

# #### That marks the end of my exploratory data analysis. In the enst session, I will look at how unsupervised machine learning can segment the customers based on the patterns in this dataset.

# # SESSION 4: KMeans Clustering

# In[33]:


# Create a copy of the dataset
x = fda.copy()


# In[34]:


# Check the categorical and numeric variables
xnum_list= []
xcat_list= []

for feat in fda:
    if is_numeric_dtype(x[feat]):
        xnum_list.append(feat)
    elif is_string_dtype(x[feat]):
        xcat_list.append(feat)
        
print("numeric: ", xnum_list)
print("Categorical: ", xcat_list)


# # Feature Selection

# ###  To build a base clustering model;
# 
# - I will drop all the categorical features and then evaluate the result.
# - Feed the categorical varables into the modle and then compare the results.
# - For the numeric variables, I prioritized the sales KPIs alone.

# In[35]:


# Drop the categorical variables;

x= x.drop(['Age Range', 'Country', 'Gender', 'Line Category', 'Line SKU', 'Order ID', 'Status','Marketing optin','New repeaters', 'New active'], axis=1)
x.head()


# # Building the model

# In[36]:


# I created an object and called it kmeans
# The number in the brackets is K, or the number of clusters I am trying aiming for

kmeans = KMeans(n_clusters= 2, random_state=540)
# Fit the data
kmeans.fit(x)

print ("Clusters are:", kmeans.labels_)
print("Inertia is: ", kmeans.inertia_)


# ## Clustering results

# In[37]:


# Create a copy of the input data
clusters = x.copy()

# Take note of the predicted clusters 
clusters['cluster_pred'] = kmeans.fit_predict(x)


# In[38]:


# Plot the clusters
plt.figure(figsize=(8,6), tight_layout=True)
plt.scatter(clusters['Net Sales'],clusters['Line SKU Production Cost'],c= clusters['cluster_pred'],cmap='viridis', alpha=0.8)
plt.xlabel('Gross Profit')
plt.ylabel('Line SKU Production Cost')
plt.title('Net Sales Versus Line SKU Production Cost Profit; k=2')


# From the plot above, two distint

# ### Using Inertia to decide the number of clusters

# In[39]:


no_of_clusters = range(2,10)
inertia = []

for f in no_of_clusters:
    kmeans = KMeans(n_clusters= f, random_state=2)
# Fit the data
    kmeans = kmeans.fit(x)
    u=kmeans.inertia_
    inertia.append(u)
    
    print ("Inertia for :", f, "clusters is:", u)


# ## Plot the inertia graph

# In[40]:


fig, (ax1)= plt.subplots(1, figsize=(8,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx, inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters, rotation='vertical')
plt.xlabel("No. of clusters")
plt.ylabel("inertia score")
plt.title("Inertia plot per k")


# #### The plot shows an elbow at 3,4 and 6 bends, so I will try k=4 and then evaluate the results

# ## Explore clustering solutions with k=4

# In[41]:


x_unscaled = x.copy() # copy original dataframe

kmeans_new = KMeans(n_clusters=4, random_state=2) # set kmeans

# Fit the data
kmeans_new.fit(x_unscaled)
kmeans_new.labels_

new_x = x_unscaled.copy() #make a copy of the fitted data

new_x['Clusters']= kmeans_new.fit_predict(x_unscaled) #predict the  clusters

# calculating the counts of clusters

unique, counts=np.unique(new_x['Clusters'], return_counts= True)
counts= counts.reshape(1,4)

#create a df of the counts

count_df= pd.DataFrame(counts,columns=["Cluster 0",
                                      "Cluster 1",
                                      "Cluster 2",                   
                                      "Cluster 3"])

count_df


# ### The result of the clustering produced 3 market segments, with cluster 0 containing ~82%. This does not look like a fair distribution, but then , the clustering algorithim has aggregated the different datapoints and produced 3 clusters based on the similarities between them.
# 
# - Next, I will visualize the cluster and also check the sillhoutte score of the clustering

# In[78]:


# Plot the clusters
plt.figure(figsize=(8,6), tight_layout=True)
plt.scatter(new_x['Net Sales'],new_x['Line SKU Production Cost'],c= new_x['Clusters'],cmap='viridis', alpha=0.8)
plt.xlabel('Line SKU Production Cost')
plt.ylabel('Net Sales')
plt.title('Net Sales Versus Line SKU Production Cost; k=4')


# #### I can clearly see the four segments, one at the left bottom, two in the middle and one at the top right.

# # Clustering Evaluation

# ### Silhoutte_Score
# 
# - The value of the silhouette coefﬁcient is between [-1, 1].
# - A score of 1 denotes the best, meaning that the data point i is very compact within the cluster to which it belongs and far away from the other clusters.
# - The worst value is -1. Values near 0 denote overlapping clusters.

# In[43]:


score = silhouette_score(new_x, kmeans_new.labels_, metric='euclidean')
print(score) 


# In[44]:


#### 0.69 is a fairly high silhouette score. However, let me try to improve the results by playing with more features.


# # Improving results with Feature Selection

# In[45]:


# Make another copy of the processed dataframe

x2 = fda.copy()
x2.head()


# ### I will drop the categorical variables, but this time, include gender, age range and the line category in the model

# In[46]:


# Drop the categorical variables;

x2= x2.drop(['Country','Line SKU', 'Order ID', 'Status','Marketing optin','New repeaters', 'New active'], axis=1)
x2.head()


# # Encoding Categorical Variables

# In[47]:


ce_one = ce.OneHotEncoder(cols=['Age Range','Line Category','Gender'])  #instantiate the one_hot encoder

x2= ce_one.fit_transform(x2)

x2.head(5)


# In[48]:


x_unscaled = x2.copy() # copy original dataframe

kmeans_new2 = KMeans(n_clusters=4, random_state=2) # set kmeans

# Fit the data
kmeans_new2.fit(x_unscaled)
kmeans_new2.labels_

new_x = x_unscaled.copy() #make a copy of the fitted data

new_x['Clusters']= kmeans_new2.fit_predict(x_unscaled) #predict the  clusters

# calculating the counts of clusters

unique, counts=np.unique(new_x['Clusters'], return_counts= True)
counts= counts.reshape(1,4)

#create a df of the counts

count_df2= pd.DataFrame(counts,columns=["Cluster 0","Cluster 1","Cluster 2","Cluster 3"])         
                                      

count_df2


# # Compare both results

# In[49]:


score_scaled = silhouette_score(x, kmeans_new2.labels_, metric='euclidean')

print(score)
print(score_scaled)
print(" ")
print(count_df)
print(count_df2)


# #### The results are exactly the same, hence the newly added varaibles didn't influence the clustering result.

# # NEXT STEPS;
# 
# #### After segmenting the customers with the KMeans algorithim, the next step will be to export the dataset and then analyze it based on the cluster categories. To make a business decison, it is necessary to identify the common traits in each segment.
# 
# ####  This is what will inform further marketing, product development and sales strategies.
# 
# #### In addition, exploring more number of clusters and other clustering algorithims might produce a better result.

# # Export dataset for analysis in Tableau

# In[50]:


clusters= kmeans_new.labels_
fda['Clusters']= clusters


# In[60]:


fda['Clusters Category']= 'No Data'
fda['Clusters Category'].loc[fda['Clusters'] == 0] = 'Cluster 1'
fda['Clusters Category'].loc[fda['Clusters'] == 1] = 'Cluster 2'
fda['Clusters Category'].loc[fda['Clusters'] == 2] = 'Cluster 3'
fda['Clusters Category'].loc[fda['Clusters'] == 3] = 'Cluster 4'

fda.head(5)


# In[61]:


fda.to_csv('fda_cluster', encoding='utf-8')


# # Explore Clustering Result

# In[62]:


fda['Clusters Category'].value_counts()


# In[63]:


#Explore the model clustering results 
np.round(fda.groupby(['Clusters Category'])[['Gross Profit','Net Sales']].describe(),1)


# In[81]:


# I am interested in how the gender categories are distributed by the clusters;

# Show gender distribution by clusters 
plt.figure(figsize=(6,6), tight_layout=True)
ax = sns.scatterplot(data= fda, x='Gender', y='Consumer LTV', alpha= 0.9 , hue='Clusters Category', palette='viridis', s=70)
ax.set(xlabel='Product Category', ylabel='Net Sales')
plt.title('Net Sales by Product Categories with respect to the Clusters')
plt.xticks(rotation=45)


# # Insights;
# 
# #### The chart above shows four segments of customers even within the same gender. 
# 
# - In both gendr categories, there is a segment that are at the very bottom <strong>(cluster 1)</strong>. That is, they often purchase products with lower net sales.
# 
# - Next, <strong>cluster 2</strong>, with a net sale between £1000- £2500.
# 
# - The third segment, <strong>cluster 3</strong>,in the female categories often purchase products with net sales from about £3500 to above £7000. Interestingly,this segment is not present in the male category.
# 
# - Lastly, the <strong>cluster 2</strong> have a bit of spread. They intersect with some members of cluster 2. However, they range from net sales £1500 to nearly £3200 in the female gender. Within the male category, the net sale sis around £2000.
# 
# 
# ### What about using a pair plot to visulaize the patterns in the numeric at once???

# In[71]:


sns.set_style('ticks')
sns.pairplot(fda,vars=['Gross Profit','Net Sales','Discount','Line SKU Production Cost'], hue='Clusters Category', diag_kind='kde', kind='scatter', palette='viridis')
plt.show()


# ### To be continued...
