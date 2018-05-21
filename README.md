# House prices: Ensemble and a comprehensive EDA (keep updating)

## Introduction

Kaggle describes this competition as follows:

Ask a home buyer to describe their dream house, and they probably won’t begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition’s dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

## Executive Summary:

I started the competition by focusing on detailed data exploration just to have a great grasp of the dataset, which is very useful for dealing with missing value and conducting effective feature engineering. EDA process will be introduced here with many visualizations. This project includes

* **Imputing missing values** by processing sequentially through each feature

*	**Transforming** some numerical variables that are actually categorical 

*	**Scaling** all numerical variables

*	**Label Encoding** the categorical variables that are ordinal

*	**Getting dummy variables** for  all categorical features

* Build Pipelines in Machine Learning

* Grid Search parameters to optimize a machine learning model

*	Lasso regression

* Random Forest

*	XGBoost

*	Stacking

## Exploratory Data Analysis

**First, Let's start with loading libraries from python**
![1](https://user-images.githubusercontent.com/38633055/40275135-0c473a28-5bb4-11e8-9ea5-3d86c37fb3f4.png)
Next, we load the dataset and store it in a dataframe called "df", from the shape of which we can see that there are 1460 records and 81 features. Our response variable is "SalePrice".
![2](https://user-images.githubusercontent.com/38633055/40275173-e5d32828-5bb5-11e8-92b6-63a024f3b4df.png)

After loading the dataset, we would like to do some visualization just to have a clear view of our dataset . The first thing I'm interested in is to see how many missing value in each feature. Because our number of features is not small, for a clear view I ordered the number of missing values in a descending sequence.
![3](https://user-images.githubusercontent.com/38633055/40275191-ba4f4e4c-5bb6-11e8-890a-74ec51254d67.png)

**SalePrice is the variable we need to predict and because variables like price in reality is often right skewed, so Let's do some analysis on it**. 

Basic statistics are shown here:
![4](https://user-images.githubusercontent.com/38633055/40275317-693e84d8-5bba-11e8-8400-569977c53993.png)

We can also use qqplot to check it's normality
![5](https://user-images.githubusercontent.com/38633055/40275327-9741b670-5bba-11e8-9078-51a972d7afc0.png)

As expected the price variable **"SalePrice"** is right skewed:
![6](https://user-images.githubusercontent.com/38633055/40275350-4e1ed0f8-5bbb-11e8-9582-bd8d98902ce1.png)
![7](https://user-images.githubusercontent.com/38633055/40275361-9c2fa3da-5bbb-11e8-86cd-5daee85a3259.png)

As we can see here, the variable "SalePrice" is right skewed and does not show normality in distribution, however most regression models would require normal distribution as their assumptions. We need to add the logrithm to transform the target variable.

Now it looks normal! Both qqplot and density plot reflect that the transformation works
![8](https://user-images.githubusercontent.com/38633055/40275470-9b462974-5bbd-11e8-919d-c66897eb8ab5.png)
![9](https://user-images.githubusercontent.com/38633055/40282234-4a522a4a-5c3a-11e8-8a87-39e9661e6356.png)

**Correlation Matrix:**

![10](https://user-images.githubusercontent.com/38633055/40282419-f2d7b732-5c3c-11e8-9a19-52693605b284.png)

The correlation matrix may still look a bit unclear to you so I extracted top 10 correlated pairs
![12](https://user-images.githubusercontent.com/38633055/40282711-6e1cad9a-5c41-11e8-8c57-7034fb9a4a49.png)

According to the correlation analysis these are the variables mostly correlate with "SalePrice". We can see that **OverallQual** is highly mostly correlated with **SalePrice** with coefficient 0.79, followed by "GrLivArea" with coefficient 0.7. Features like "GarageCars", "GarageArea" and "TotalBsmtSF" are less but still highly correlated with prediction variable with coeffient at 0.64, 0.62 and 0.61. Since the total 5 variables are all highly correlated with house price, multicolinearity may exist.

**Pairwise plot with houseprice:**
![13](https://user-images.githubusercontent.com/38633055/40316986-0ab92f1e-5cee-11e8-9300-717b41c14b49.png) 
![14](https://user-images.githubusercontent.com/38633055/40317241-e84d242a-5cee-11e8-8a4c-eb7d91c37b72.png)

From figures above it seems that overall quality of the house and house price have strong linear relation. "OverallQual" is an orderal variable ranked from 1 to 10. 10 means highest quality which makes sense as normally the higher the quality the higher the price. The linear correlation between "GrLivArea"(ground living area square feet) and house price is less obvious and most datapoints are concentrated at "GrLivArea"=[1000,3000]. "GrLivArea" is the second highest correlated variable which makes sense as expensive houses should have big living area

![15](https://user-images.githubusercontent.com/38633055/40317293-0854ed70-5cef-11e8-82e0-210736767725.png)

It seems "GarageCars"(size of garage in car capacity) and house price have great linear correlation. "GarageCarS" is an orderal variable ranked from 0 to 4. 4 means biggest size which in general makes sense as usually it takes higher price to have a bigger garage. However we also notice that most size of garage is concentrated between 0 and 3, and we don't see such linear correlation in size value 4. Also there are more outliers in size 3 and 4 than smaller size.

## Feature Engineering
Next I will do some feature engineering to the existing dataset. Since in reality even structured datasets will have some missing value, not clean data formating issues, and need further modification. This section will therefore include imputing missing data, re-defining categorical & numerical variables, dropping existing irrelavant features, or adding & merging new features.

### Missing Values
1



















