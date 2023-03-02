# Fashion-Data-Analytics-Market-Segmentation-with-KMeans-Clustering![Fashion Market Segmentation](https://user-images.githubusercontent.com/71575857/222194112-3c7b9f92-7636-4b9a-834f-993ee39f47d6.png)


Aim: 
The aim of this project is to look at customers in-depth and make a machine-learning model that can put them into groups without being watched.
 
Application Used: 
Python Language

About the datasets:
The dataset used for this project is a fashion retail dataset from 365 Datascience. It has a total of 31 columns and 3,675 rows.

Methodology: 
The first thing I did was check the missing values, confirm the data types, and identify the variables that I will need for this project.
  
 - Exploratory Data analysis: After the data cleaning, I did an in-depth data analysis using descriptive statistics, box plots, count plots, scatter plots and pair plots to gain a better understanding of the customer information. 

Please see the notebook for the data insights.
 
Clustering: 
For the clustering, I had created two groups of features ;
Group 1 (numeric variables alone) :
- Average order value
- Consumer LTV
- Discount
- Gross Profit
- Line SKU Production Cost
- Net Sales
- Total order value
Group 2:
I used the features in the group1 and added some categorical variables ;
- Gender
- Age Range
- Line Category.
 
Deciding the number of clusters:
Elbow Method
 
 Model Evaluation:
Sillouette score evaluate the clustering.
 
Clustering Insights
 
In both gender categories, there is a segment that are at the very bottom <strong>(cluster 1)</strong>. That is, they often purchase products with lower net sales.
- Next, cluster 2, with a net sale between £1000- £2500.
- The third segment,cluster 3,in the female categories often purchase products with net sales from about £3500 to above £7000. Interestingly,this segment is not present in the male category.
- Lastly, the cluster 2</ have a bit of spread. They intersect with some members of cluster 2. However, they range from net sales £1500 to nearly £3200 in the female gender. Within the male category, the net sale sis around £2000.
 
Further Work:
  
The next phase of the project will be to do another exploratory data analysis using the clusters/segments and creating a #tableau #dashboard.
