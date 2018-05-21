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

![10](https://user-images.githubusercontent.com/38633055/40322961-fd4b7e54-5d01-11e8-852e-3b92b56c7f77.png)

The correlation matrix may still look a bit unclear to you so I extracted top 10 correlated pairs
![12](https://user-images.githubusercontent.com/38633055/40282711-6e1cad9a-5c41-11e8-8c57-7034fb9a4a49.png)

According to the correlation analysis these are the variables mostly correlate with "SalePrice". We can see that **OverallQual** is highly mostly correlated with **SalePrice** with coefficient 0.79, followed by "GrLivArea" with coefficient 0.7. Features like "GarageCars", "GarageArea" and "TotalBsmtSF" are less but still highly correlated with prediction variable with coeffient at 0.64, 0.62 and 0.61. Since the total 5 variables are all highly correlated with house price, multicolinearity may exist.

**Pairwise plot with houseprice:**
![13](https://user-images.githubusercontent.com/38633055/40316986-0ab92f1e-5cee-11e8-9300-717b41c14b49.png) 
![14](https://user-images.githubusercontent.com/38633055/40317241-e84d242a-5cee-11e8-8a4c-eb7d91c37b72.png)

From figures above it seems that overall quality of the house and house price have strong linear relation. "OverallQual" is an orderal variable ranked from 1 to 10. 10 means highest quality which makes sense as normally the higher the quality the higher the price. The linear correlation between "GrLivArea"(ground living area square feet) and house price is less obvious and most datapoints are concentrated at "GrLivArea"=[1000,3000]. "GrLivArea" is the second highest correlated variable which makes sense as expensive houses should have big living area. **Here you might already noticed that the two data points at the right bottom corner seem counterintuitive. It's unreasonable for these two houses having very big living area (around 5000 square feet) but very low price.** I dropped these two data points later and this step do improve my final model error!

![15](https://user-images.githubusercontent.com/38633055/40317293-0854ed70-5cef-11e8-82e0-210736767725.png)

It seems "GarageCars"(size of garage in car capacity) and house price have great linear correlation. "GarageCarS" is an orderal variable ranked from 0 to 4. 4 means biggest size which in general makes sense as usually it takes higher price to have a bigger garage. However we also notice that most size of garage is concentrated between 0 and 3, and we don't see such linear correlation in size value 4. Also there are more outliers in size 3 and 4 than smaller size.

## Feature Engineering
Next I will do some feature engineering to the existing dataset. Since in reality even structured datasets will have some missing value, not clean data formating issues, and need further modification. This section will therefore include imputing missing data, re-defining categorical & numerical variables, dropping existing irrelavant features, or adding & merging new features.

### Missing Values
Usually dataset will contain a lot of missing data. Some of them are not really "missing" just because they are none in nature.For example, quality measurements of a garage can have **NA** when there is no garage. In this case there is no need to impute the missing data. However there are other types of missing value requiring filling in order to maintain effectiveness of data.

First of all, I would like to see which variables have missing value and how many are missing:

![16](https://user-images.githubusercontent.com/38633055/40322576-cb566126-5d00-11e8-9502-8b6e7c7cf76d.png)

So there are 19 variables having missing data and I'm going to deal with them one by one. 
#### PoolQC
As its name suggests, this is quality of the pool of houses. Based on data description, NA simply implies there is no pool in the house.
![17](https://user-images.githubusercontent.com/38633055/40323413-4a67e514-5d03-11e8-9615-7d241a4cfeff.png)

#### MiscFeature
This indicates miscellaneous feature not covered in other categories. It is a categorical variable and to see it's impact on house price I apply box plot to check
![18](https://user-images.githubusercontent.com/38633055/40323953-01a00986-5d05-11e8-9e19-31d1cb835bb0.png)
![19](https://user-images.githubusercontent.com/38633055/40323850-bc9cf7c2-5d04-11e8-8563-1610ec834ddf.png)

I decided to drop this variable because I don't see pattern in this box plot and this type of irrelavant variable will distract final model fitting

#### Alley
This indicates type of alley access to property. It is categorical so we still use boxplot to check:
![20](https://user-images.githubusercontent.com/38633055/40324215-dded8792-5d05-11e8-92a6-856bfe6ba4d7.png)
![21](https://user-images.githubusercontent.com/38633055/40324370-538a2fa0-5d06-11e8-9fc1-1877ca77de41.png)

NA of this variable means that there is no alley access so I assign string "other" to replace NA. As suggested in the boxplot(the two in the middle of above figure), type "Gravel" and type "Paved" seem to have very similar impact so I assign and aggregate both "Grvl" and "Pave" as "Both"

#### Fence
This indicates fence quality and like most quality measurement, NA implies no fence. "Fence" is also a categorical variable I still use boxplot to check if we need to keep this variable. In order to do this I need to see how many distinct value in this feature:
![22](https://user-images.githubusercontent.com/38633055/40325028-62dad9d0-5d08-11e8-8d47-bfee508414cf.png)
![23](https://user-images.githubusercontent.com/38633055/40325580-4da7e010-5d0a-11e8-82ac-5358ad9d7e40.png)

All levels seem stable and even, which means fence quality doesn't have noticable impact on house price, so I dicided to drop this feature

#### FireplaceQu
This indicates fireplace quality and similarly NA means no fireplace, so I replace NA with "None". Again, we use boxplot to check it's impact on house price
![25](https://user-images.githubusercontent.com/38633055/40325978-b2f07dc8-5d0b-11e8-84d0-295a660b8a1e.png)
![24](https://user-images.githubusercontent.com/38633055/40325935-8c8594e8-5d0b-11e8-953b-6f84dfb72b39.png)
![34](https://user-images.githubusercontent.com/38633055/40333973-30dd52fc-5d29-11e8-9616-99a9937804d3.png)

As suggested by the plot above the boxes with better quality have higher sale price, which makes sense. Even though there are some outliers in some categories(the left three boxes), I still decided to keep this variable.

#### LogFrontage
This means linear feet of street connected to property. I decided to drop this because the correlation coeffcient to sale price is low and linear plot also shows weak correlation

![26](https://user-images.githubusercontent.com/38633055/40326503-79a2d244-5d0d-11e8-9e68-359dd2a4c19e.png)

#### GarageCond
As it's name suggested this means the condition of the garage. This is also a categorical variable so I use boxplot to visualize the data. I decided to drop this because all levels seem on a horizontal line and this variable doesn't contribute much information to sale price

![27](https://user-images.githubusercontent.com/38633055/40326794-8b0247f8-5d0e-11e8-9bf1-c2ec73755377.png)

#### Other Garage-Related Variables: GarageType, GarageYrBlt, GarageFinish, GarageQual
These four garage-related variables are very similar and NA indicates no garage. So I replaced NA with string "None"
![28](https://user-images.githubusercontent.com/38633055/40327044-6ea67240-5d0f-11e8-8abf-fd1cca8476ee.png)

#### Basement-related Variables:
Similarly to garage-related variables, these five basement-related variables are very similar and NA indicates no basement. So I replaced NA with string "None"
![29](https://user-images.githubusercontent.com/38633055/40327275-4699438a-5d10-11e8-905f-20cbc206b32c.png)

#### MasVnrType
According to the feature definition, this is masonry veneer type and NA means there is no masonry. Since this is also a categorical variable the boxplot is applied. I will keep this variable and fill NA with "None"
![30](https://user-images.githubusercontent.com/38633055/40333550-0c67d264-5d27-11e8-8f8b-34069b85630c.png)
![31](https://user-images.githubusercontent.com/38633055/40333566-27dba958-5d27-11e8-92f1-a342373aadab.png)

#### MasVnrArea
This variable denotes masonry veneer area in square feet and similar to MasVnrType NA means none masonry actually. This is a numerical variable and in order to better visualize it, I remove the missing value.
![33](https://user-images.githubusercontent.com/38633055/40333722-f42973d2-5d27-11e8-9c68-03ce9e8b3436.png)

We can see that linear correlation exists and I fill the missing value with the average of none-NA value.

#### Electrical
This variable represents electrical system.Since we only have 1 missing value in this variable, I replace the NA with the highest-frequency category. 
![35](https://user-images.githubusercontent.com/38633055/40334890-a29ab65a-5d2e-11e8-9e78-050aa71a9f2a.png)

The most frequent electrical system is "SBrkr" with frequency 1334.

Up to now we have filled with all missing value!
![38](https://user-images.githubusercontent.com/38633055/40335194-6ddb68fe-5d30-11e8-85e8-08b0139d9ae8.png)













