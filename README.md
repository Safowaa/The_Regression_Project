  # Time Series Analysis Corporacion Favorita
In this project, we conduct a comprehensive time series analysis of sales data from a chain of stores for Corporacion Favorita. 



## Project Overview

The aim is to develop a model that accurately forecasts the unit sales of thousands of items across various Favorita stores.The analysis will employ various forecasting models, such as ARIMA, ETS, RandomForest, and XGBoost, to predict future sales trends. Through this study, we intend to provide actionable insights for optimizing inventory management and marketing strategies. Ultimately, the findings will support the stores in enhancing their operational efficiency and profitability.


**Table of contents**<a id='toc0_'></a>    
- [The Regresson Project](#toc1_)    
  - [Business Understanding](#toc1_1_)    
  - [Background](#toc1_2_)      
  - [Scenario](#toc1_3_)          
  - [Methodology](#toc1_5_)    
  - [Additional Notes](#toc1_6_)     
  - [Hypothesis](#toc1_7_) 
  - [Analytical Questions](#toc1_8_)    
  - [Data Understandng](#toc1_9_)    
    - [Data Collection](#toc1_9_1_)  
    - [`Importation`](#toc1_9_2_)    
    - [` Data Loading `](#toc1_9_3_)    
  - [Exploratory Data Analysis: EDA](#toc1_10_)    
    - [Visualizing Holidays and Events](#toc1_10_1_)    
      - [Yearly Trends:](#toc1_10_1_1_)    
    - [Insights pertaining to stores](#toc1_10_2_)   
  - [`Univariate Analysis`](#toc1_11_)    
  - [`Bivariate & Multivariate Analysis`](#toc1_12_)    
  - [Feature Processing & Engineering](#toc1_13_)    
  - [Answering Hypothesis Questions](#toc1_14_)    
    - [Deduction](#toc1_14_1_)    
    - [Conclusion:](#toc1_14_2_)    
  - [Answering Analytical Questions](#toc1_15_)   
  - [Machine Learning Processes And Models](#toc1_16_)    
      - [Pipeline Creation](#toc1_16_1_1_)    
      - [Data Spliting](#toc1_16_1_2_)    
    - [Feature Engineering and Scaling](#toc1_16_2_)    
    - [Define models with batch processing](#toc1_16_3_)    
    - [Model Creation](#toc1_16_4_)    
- [Initialize dictionaries to store predictions](#toc2_)    
- [Train and evaluate models](#toc3_)    
    - [Functions to calculate evaluation metrics](#toc3_1_1_)    
    - [Evaluating Models](#toc3_1_2_)    
      - [**Evaluating ARIMA**](#toc3_1_2_1_)    
      - [**Evaluating RANDOMFOREST**](#toc3_1_2_2_)    
      - [**Evaluating XGBOOST**](#toc3_1_2_3_)    
      - [**Evaluating EXPONENTIALSMOOTHING**](#toc3_1_2_4_)    
- [Export Best Performing Model](#toc4_)    

<!-- vscode-jupyter-toc-config
	numbering=false
	anchor=true
	flat=false
	minLevel=1
	maxLevel=6
	/vscode-jupyter-toc-config -->
<!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->
## Introduction

Time series analysis can help stores anticipate sales spikes during holidays and major events, reducing the risk of stockouts. By accurately forecasting demand, stores can ensure they have sufficient inventory to meet customer needs. This proactive approach enhances customer satisfaction and maximises sales opportunities.

## Data Description

The dataset used in this project contains various attributes:

- id
- date
- store_nbr
- family
- sales
- onpromotion
- day
- month
- week
- year
- city
- state
- store_type
- cluster	
- transactions	
- dcoilwtico	
- holiday_type	
- locale	
- locale_name	
- description	
- transferred

## Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution of data, detect anomalies, and identify relationships between features. Key findings include:

- Distribution of customers based on tenure, monthly charges, and contract type.
### EDA Visualization
#### Holidays and Events
![Holidays and Events](https://github.com/Safowaa/The_Regression_Project/blob/master/Visuals/H%26E.png)

![Oil Sales](https://github.com/Safowaa/The_Regression_Project/blob/master/Visuals/OIL.png)

![Sales Overtime](https://github.com/Safowaa/The_Regression_Project/blob/master/Visuals/SALES.png)


## Some Business Questions and Visualizations

Several business questions were asked and answered through visualizations:

### PowerBI Dashboard 
![Telco Churn Analysis Dashboard](https://github.com/Safowaa/The_Regression_Project/blob/master/Visuals/PowerBI.png)

1. **Compare the sales for each month across the years and determine which month of which year had the highest sales.**
   ![Year and it highest month sale](https://github.com/Safowaa/The_Regression_Project/blob/master/Visuals/m%26s.png)
   
2. **Did the earthquake impact sales?**
   ![Service to Churn](https://github.com/Safowaa/The_Regression_Project/blob/master/Visuals/sale%26earthquake.png)
   
3. **Are sales affected by promotions, oil prices and holidays?**
   ![Correlation Heatmap of oil price, promotions, and holidays](https://github.com/Safowaa/The_Regression_Project/blob/master/Visuals/promo%26oil.png).

4. **What analysis can we get from the date and its extractable features?**
   ![.](https://github.com/Safowaa/The_Regression_Project/blob/master/Visuals/cluster1.png)
## Machine Learning Model

A machine learning model was built to predict future sales trends and optimise inventory management for the store. The process involved:

- Data preprocessing: Handling missing values, encoding categorical variables, and feature scaling.
- Model selection: Evaluated multiple models including ARIMA, RandomForest, XGBoost etc.
- Model evaluation: Used metrics such as rmsle, mse, rmse, mae to evaluate model performance.

The best-performing model was chosen based on these evaluation metrics and used to predict the churn column in the test data.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Safowaa/The_Regression_Project.git



## Conclusion

The time series analysis and machine learning model have provided valuable insights into the store's sales patterns, helping to anticipate demand fluctuations and enhance decision-making. By accurately predicting sales, especially during holidays and major events, the store can optimise its inventory and reduce stockouts, ultimately improving customer satisfaction and profitability. Implementing these predictive strategies will enable the store to be more agile and responsive to market changes, ensuring sustained growth and success.

## Recommendations

Stores are encouraged to save data in a more stuctured way.

## References

- Visit [Link](https://app.powerbi.com/groups/me/reports/08def0ef-9c0f-49f7-a050-7415d32320c2/ReportSection?experience=power-bi) to interact with the dashboard of this project.

- Visit [Link](https://medium.com/@safowaabenedicta/the-regression-project-008d6d90981b) to read article.
