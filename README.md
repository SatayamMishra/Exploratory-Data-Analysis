# Name: Satyam Mishra

# Task 3: Exploratory Data Analysis

# GRIP@SparkFoundation - Data Science and Business Analytics - JUNE2022

# Objective

● Perform ‘Exploratory Data Analysis’ on dataset ‘SampleSuperstore’

● As a business manager, try to find out the weak areas where you can work to make more profit.

● What all business problems you can derive by exploring the data?


# Import the all required libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans

# Reading the data 

dataset=pd.read_csv("C:/csvfile/SampleSuperstore.csv")
print("Data Import Succesfully")
dataset.head(5)

dataset.shape

dataset.info()

dataset.describe()

# Checking missing value

dataset.isnull().sum()

# Checking the duplicates values

dataset.duplicated().sum()

# Deleting the duplicates value

dataset.drop_duplicates()

dataset.nunique()

# Deleting the Variable

column=['Postal Code']
dataset1=dataset.drop(columns=column,axis=1)

# Correlation between Variable

dataset1.corr()

dataset1.head()

# Exploratory Data Analysis

# Data Visualization

plt.figure(figsize=(16,8))
plt.bar("Sub-Category","Category",data=dataset1)
plt.title("Category Vs Sub-Category",color="Green")
plt.xlabel("Sub-Category",color="Green")
plt.ylabel("Category",color="Green")
plt.xticks(rotation=40)
plt.show()

# Plotting Histogram

dataset1.hist(bins=50,figsize=(15,10))
plt.show()
print("From the Seing of Histogram Graph We can say that Our Data is not Normal")

# Count the total Repeatable States

dataset1["State"].value_counts()

plt.figure(figsize=(15,10))
sns.countplot(x=dataset1["State"])
plt.xticks(rotation=70)
plt.title("State",color="Green",fontsize=18)
plt.show()

Next,"Copiers" Sub-category has gain highest amount of profit with no loss.There are other sub-categories too who are not faced any kind of losses but their profit margins are also low.

# Next,Suffering from highest loss is machines

sns.set(style="darkgrid")
plt.figure(2,figsize=(15,10))
sns.barplot(x="Sub-Category",y="Profit",data=dataset1,palette="Spectral")
plt.suptitle("Pie Conjuption Patterns in the United States",fontsize=18,color="Green")
plt.show()

figsize=(15,10)
sns.pairplot(dataset1,hue="Sub-Category")
plt.show()

### ● From the above plot we can say that Our Data is not Normal and it has some amount of outliers too.

### ● Let's explore more about these outliers by using boxplots.

### ● Ist we'll check Sales from Every Segments of Whole Data

From above Graph we can say that "Home Office" segment has less purchased Sub-Categories and in that "Tables","Supplies","Machines","Copiers","Bookcases" has the lowest Sales. "Consumer" has purchased more sub-categories as compared to other segments.

sns.lineplot('Discount','Profit', data=dataset1,color='r',label='Discount')
plt.legend()
plt.figure(figsize=(10,4))
plt.show()

state_code = {'Alabama': 'AL','Alaska': 'AK','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO',
              'Connecticut': 'CT','Delaware': 'DE','Florida': 'FL','Georgia': 'GA','Hawaii': 'HI','Idaho': 'ID',
              'Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA',
              'Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi'
              : 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ',
              'New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Ohio': 'OH','Oklahoma': 'OK',
              'Oregon': 'OR','Pennsylvania': 'PA','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD',
              'Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virginia': 'VA','District of Columbia': 'WA',
              'Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'}
dataset1['state_code'] = dataset1.State.apply(lambda x: state_code[x])

state_data=dataset1[['Sales', 'Profit', 'state_code']].groupby(['state_code']).sum()

fig=go.Figure(data=go.Choropleth(
    locations=state_data.index, 
    z = state_data.Sales, 
    locationmode = 'USA-states', 
    colorscale = 'Reds',
    colorbar_title = 'Sales in USD',))

fig.update_layout(
    title_text = 'Total State-Wise Sales',
    geo_scope='usa',
    height=400,)

fig.show()

Now, let us analyze the sales of a few random states from each profit bracket (high profit, medium profit, low profit, low loss and high loss) and try to observe some crucial trends which might help us in increasing the sales.

# We have a few questions to answer here.

#1 What products do the most profit making states buy?

#2 What products do the loss bearing states buy?

#3 What product segment needs to be improved in order to drive the profits higher?


def state_data_viewer(states):
    """Plots the turnover generated by different product categories and sub-categories for the list of given states.
    Args:
        states- List of all the states you want the plots for
    Returns:
        None
    """
    product_data = dataset1.groupby(['State'])
    for state in states:
        data = product_data.get_group(state).groupby(['Category'])
        fig, ax = plt.subplots(1, 3, figsize = (28,5))
        fig.suptitle(state, fontsize=14)        
        ax_index = 0
        for cat in ['Furniture', 'Office Supplies', 'Technology']:
            cat_data = data.get_group(cat).groupby(['Sub-Category']).sum()
            sns.barplot(x = cat_data.Profit, y = cat_data.index, ax = ax[ax_index])
            ax[ax_index].set_ylabel(cat)
            ax_index +=1
            
fig.show()

states = ['California', 'Washington', 'Mississippi', 'Arizona', 'Texas']
state_data_viewer(states)

# Now We Clustering Analysis(K-Means Cluatering)

x=dataset1.iloc[:,8:12]

wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0).fit(x)
    wcss.append(kmeans.inertia_)

sns.set_style("darkgrid") 
sns.FacetGrid(dataset1, hue ="Sub-Category",height = 5).map(plt.scatter,'Sales','Quantity')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'brown', label = 'Centroids')

plt.legend()
plt.show()

sns.set_style("darkgrid") 
sns.FacetGrid(dataset1, hue ="Sub-Category",height = 5).map(plt.scatter,'Sales','Profit')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'brown', label = 'Centroids')

plt.legend()
plt.show()

fig,ax=plt.subplots(figsize=(10 , 6))
ax.scatter(dataset1["Sales"] , dataset1["Profit"])
ax.set_xlabel('Sales',color='green',fontsize=16)
ax.set_ylabel('Profit',color='green',fontsize=16)
ax.set_title('Sales vs Profit',color='green',fontsize=18)
plt.show()

# Finding

● Profit is more than that of sale but there are some areas where profit could be increased

● Profit and Discount is high in First Class

● Sales is high for Same day ship

#### ● Here is top 3 city where deals are Highest

1.New York City

2.Los Angeles

3.Philadelphia

#### ● Here is top 3 city where deals are Highest

1.New York City

2.Los Angeles

3.Philadelphia
