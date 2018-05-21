# House prices: Ensemble and a comprehensive EDA (keep updating)

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

First, Let's start with loading libraries from python
![1](https://user-images.githubusercontent.com/38633055/40275135-0c473a28-5bb4-11e8-9ea5-3d86c37fb3f4.png)
Next, we load the dataset and store it in a dataframe called "df", from the shape of which we can see that there are 1460 records and 81 features. Our response variable is "SalePrice".
![2](https://user-images.githubusercontent.com/38633055/40275173-e5d32828-5bb5-11e8-92b6-63a024f3b4df.png)

After loading the dataset, we would like to do some visualization just to have a clear view of our dataset . The first thing I'm interested in is to see how many missing value in each feature. Because our number of features is not small, for a clear view I ordered the number of missing values in a descending sequence.
![3](https://user-images.githubusercontent.com/38633055/40275191-ba4f4e4c-5bb6-11e8-890a-74ec51254d67.png)

**SalePrice** is the variable we need to predict and because variables like price in reality is often right skewed, so Let's do some analysis on it. 

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

Correlation Matrix:

![10](https://user-images.githubusercontent.com/38633055/40282419-f2d7b732-5c3c-11e8-9a19-52693605b284.png)

The correlation matrix may still look a bit unclear to you so I extracted top 10 correlated pairs
![12](https://user-images.githubusercontent.com/38633055/40282711-6e1cad9a-5c41-11e8-8c57-7034fb9a4a49.png)
















