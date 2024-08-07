# %% [markdown]
# # The Regresson Project

# %% [markdown]
# ## Business Understanding

# %% [markdown]
# ## Background

# %% [markdown]
# #### This project centres on time series forecasting to predict store sales for Corporation Favorita, a major Ecuadorian-based grocery retailer. The aim is to develop a model that accurately forecasts the unit sales of thousands of items across various Favorita stores.

# %% [markdown]
# ## Scenario

# %% [markdown]
# #### As the team lead for data science at Corporation Favorita, a large Ecuadorian-based grocery retailer, my goal is to ensure that we always have the right quantity of products in stock. To achieve this, we have decided to build a series of machine learning models to forecast the demand for products in various locations. The marketing and sales team have provided us with some data to aid in this endeavour. Our team uses the CRISP-DM Framework for our Data Science projects.

# %% [markdown]
# ## Objective

# %% [markdown]
# ### This project will focus on two key areas.
# 
# #### *Understanding the Data:* The first objective is to gain insights into the store sales data, including store-specific information, product families, promotions, and sales numbers. This understanding will enable the company to make informed business decisions.
# 
# #### *Predicting Store Sales:* The goal is to develop a reliable time series forecasting model that accurately predicts the unit sales for different product families at various Favorita stores. This will help the company optimize inventory management, plan promotions, and improve overall sales performance.

# %% [markdown]
# ## Methodology 

# %% [markdown]
# #### *Data Exploration:* Our team will begin with a thorough exploration of the data. This involves examining the dataset to understand its structure, identifying missing values, and assessing the overall quality of the data. We will perform descriptive statistics to summarise the main features of the data and use visualisation techniques to uncover patterns and trends. By understanding the relationships between different variables, such as store-specific information, product families, promotions, and sales numbers, we will be able to identify key factors that influence sales.

# %% [markdown]
# #### *Data Preparation:* Once we have a good grasp of the data, we will move on to the data preparation phase. This step includes cleaning the data by handling missing values, outliers, and any inconsistencies. We will also perform feature engineering to create new variables that can enhance the model's predictive power. For instance, we might create features that capture seasonality, promotional effects, and store-specific trends. The data will be split into training and testing sets to validate our models effectively.

# %% [markdown]
# #### *Time Series Analysis:* With the prepared data, we will conduct a time series analysis to model and forecast sales. We will evaluate each model's performance using appropriate metrics and cross-validation techniques to ensure robustness. The chosen model will be fine-tuned to optimise its accuracy in predicting unit sales for various product families across different Favorita stores. This forecasting model will then be used to help the company optimise inventory management, plan promotions, and improve overall sales performance.

# %% [markdown]
# ## Additional Notes

# %% [markdown]
# #### **1.** Wages in the public sector are paid every two weeks on the 15th and on the last day of the month. Supermarket sales could be affected by this.

# %% [markdown]
# #### **2.** A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.

# %% [markdown]
# ## Hypothesis

# %% [markdown]
#  **`Null Hypothesis`**

# %% [markdown]
# #### There is no significant relationship between the date and the sales figures in the dataset.

# %% [markdown]
# **` Alternative Hypothesis `**

# %% [markdown]
# #### There is a significant relationship between the date and the sales figures in the dataset.

# %% [markdown]
# ## Analytical Questions

# %% [markdown]
# #### **1.** `Is the train dataset complete (has all the required dates)?`

# %% [markdown]
# #### **2.** `Which dates have the lowest and highest sales for each year (excluding days the store was closed)?`

# %% [markdown]
# #### **3.** `Compare the sales for each month across the years and determine which month of which year had the highest sales.`
# 

# %% [markdown]
# #### **4.** `Did the earthquake impact sales?`

# %% [markdown]
# #### **5.** `Are certain stores or groups of stores selling more products? (Cluster, city, state, type)`

# %% [markdown]
# #### **6.** `Are sales affected by promotions, oil prices and holidays?`

# %% [markdown]
# #### **7.** `What analysis can we get from the date and its extractable features?`
# 

# %% [markdown]
# #### **8.** `Which product family and stores did the promotions affect.`
# 

# %% [markdown]
# #### **9.** `What is the difference between RMSLE, RMSE, MSE (or why is the MAE greater than all of them?)`
# 

# %% [markdown]
# #### **10.** `Does the payment of wages in the public sector on the 15th and last days of the month influence the store sales.`

# %% [markdown]
# ## Business Questions

# %% [markdown]
# **1** ` How do sales trends vary over different seasons of the year?`

# %% [markdown]
# **2** ` What is the impact of promotions on the sales of different product families?`

# %% [markdown]
# **3** ` Are there significant differences in sales performance across different store clusters?`

# %% [markdown]
# **4** ` How do external factors, such as oil prices (dcoilwtico), affect sales trends?`

# %% [markdown]
# **5** ` What is the effect of holiday periods or special events on sales volumes? `

# %% [markdown]
# ## Data Understandng

# %% [markdown]
# ### Data Collection

# %% [markdown]
# #### The task involves accessing three distinct datasets from different sources: a database, OneDrive, and a GitHub repository. Each dataset requires a specific access method. For the database, we will query using an ODBC or ORM library. For OneDrive, we will download the file programmatically using the requests library. For the GitHub repository, we will either clone or download the file using GitPython or the requests library.

# %% [markdown]
# ### `Importation`

# %%
# Importing necessary libraries
from dotenv import dotenv_values
import pyodbc
import requests
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
import calendar
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import linregress
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVC
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense
import sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import random
import joblib
import warnings

# Suppressing all warnings to avoid cluttering the output
warnings.filterwarnings("ignore")

# Set display options for Pandas DataFrame
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)


# %%
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# %% [markdown]
# ### ` Data Loading `
# 

# %%
# Load environment variables from .env file into a dictionary
environment_variables = dotenv_values('.env')

# Get the values for the credentials you set in the '.env' file
server = environment_variables.get("server")
database = environment_variables.get("database")
username = environment_variables.get("user")
password = environment_variables.get("password")

# %%
# Create a connection string
connection_string = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"

# %%
# Use the pyodbc library to pass in the connection string.

connection = pyodbc.connect(connection_string)

# %%
# Accessing the data from the database
query = "SELECT * FROM dbo.oil"

# %%
# Viewing the table in the database 
train_1 = pd.read_sql(query, connection)
train_1

# %%
# Accessing the data from the database 
query ="SELECT * FROM dbo.holidays_events"

# %%
# Viewing the table in the database 
train_2 = pd.read_sql(query, connection)
train_2

# %%
# Accessing the data from the database 
query ="SELECT * FROM dbo.stores"

# %%
# Viewing the table in the database 
train_3 = pd.read_sql(query, connection)
train_3

# %%
# Close the database connection
connection.close()

# %%
# Read the remaining dataset in the csv file
train_4 = pd.read_csv(r"C:\Users\Safowaa\Documents\Azibiafrica\AzubiPython\The_Regression_Project\Project_data\train.csv")
train_4

# %%
# Read the remaining dataset in the csv file
train_5 = pd.read_csv(r"C:\Users\Safowaa\Documents\Azibiafrica\AzubiPython\The_Regression_Project\Project_data\transactions.csv")
train_5

# %%
# Read the remaining dataset in the csv file
train_6 = pd.read_csv(r"C:\Users\Safowaa\Documents\Azibiafrica\AzubiPython\The_Regression_Project\Project_data\test.csv")
train_6

# %%
# Read the remaining dataset in the csv file
train_7 = pd.read_csv(r"C:\Users\Safowaa\Documents\Azibiafrica\AzubiPython\The_Regression_Project\Project_data\test.csv")
train_7 

# %% [markdown]
# ## Exploratory Data Analysis: EDA

# %%
# Check the datatypes and number of columns
train_1.info()

# %%
# Check the datatypes and number of columns

train_2.info()

# %%
# Check the datatypes and number of columns

train_3.info()

# %%
#  check the datatypes and number of columns

train_4.info()

# %%
# Check the datatypes and number of columns

train_5.info()

# %%
# check the datatypes and number of columns

train_6.info()

# %%
# Check the datatypes and number of columns 

train_7.info()

# %%
#  Change the datatype of the "date" column
def convert_date_and_info(df, date_column= 'date'):
    """
    Convert a specified column of a DataFrame to datetime and print DataFrame info.

    Parameters:
    - df: pandas DataFrame containing the data.
    - date_column: str, the name of the column to convert to datetime.

    Returns:
    - None, but modifies the DataFrame in place.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    return df
convert_date_and_info(train_1)

# %%
# Change the datatypes of the "date" column

convert_date_and_info(train_2, 'date')

# %%
#  Change the datatype for the "date" column 

convert_date_and_info(train_4, 'date')

# %%
# Change the datatype of the "date" column
convert_date_and_info(train_5, 'date')


# %%
# change the datatype of the "date" column
convert_date_and_info(train_6, 'date')

# %%
# Change the datatype of the "date" column

convert_date_and_info(train_7, 'date')

# %%
def add_date_parts(df, date_column='date'):
    """
    Add day names, month, week, and year as separate columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the new columns will be added.
    date_column (str): The name of the column containing date values.

    Returns:
    pd.DataFrame: The DataFrame with additional columns for day names, month, week, and year.
    """
    df['day'] = df[date_column].dt.day_name()
    df['month'] = df[date_column].dt.month
    df['week'] = df[date_column].dt.isocalendar().week.astype(int)
    df['year'] = df[date_column].dt.year
    return df

# Create a copy of the original Dataframe to preserve the orginal data

train_4_copy = train_4.copy()
train_4_copy = add_date_parts(train_4_copy, 'date')

# %%
def analyze_dataframes(dataframes):
    """
    Analyze the missing values and duplicated rows for each DataFrame in the given dictionary.

    Parameters:
    dataframes (dict): A dictionary where keys are DataFrame names and values are DataFrames.

    Returns:
    None
    """
    for name, df in dataframes.items():
        # Determine the proportion of missing values
        missing_percentage = (df.isnull().mean() * 100).round(2)

        # Check for duplicated values
        duplicate_num = df.duplicated(subset=None, keep=False).sum()

        # Display duplicated rows if any
        duplicated_rows = df[df.duplicated(subset=None, keep=False)]

        # Display results
        print(f"Proportion of missing values in {name} dataset:")
        print(missing_percentage)
        print(f"\nNumber of duplicated rows in {name} dataset: {duplicate_num}")
        if duplicate_num > 0:
            print(f"\nDuplicated rows in the {name} dataset:")
            print(duplicated_rows)
        print("\n" + "-" * 50 + "\n")
    return dataframes
# Create a dictionary of the DataFrames to process
dataframes = {
    "train_1": train_1,
    "train_2": train_2,
    "train_3": train_3,
    "train_5": train_5,
    "train_6": train_6,
    "train_7": train_7,
    "train_4_copy": train_4_copy,
}

# Call the function to analyze the DataFrames
analyze_dataframes(dataframes)

# %%
# View columns in train_4_copy

train_4_copy.columns

# %%
# Display unique values of train_4_copy

train_4_copy.nunique()

# List the unique values in each column

for column in train_4_copy.columns:
    print(f'{column}: {train_4_copy[column].unique()}')


# %%
# Describe the numeric columns of the train_4_copy

train_4_copy.describe().T

# %%
# Describe the train_4_copy data including objects

train_4_copy.describe(include='object').T

# %% [markdown]
# ### Visualizing Holidays and Events

# %%
# Convert 'date' to datetime format if not already done
train_2['date'] = pd.to_datetime(train_2['date'])

# Extract the year from the date
train_2['year'] = train_2['date'].dt.year

# Count the number of holidays/events per year per locale
train_2_count = train_2.groupby(['year', 'locale']).size().reset_index(name='counts')

plt.figure(figsize=(15, 10))
sns.barplot(data=train_2_count, x='year', y='counts', hue='locale', palette='magma')
plt.title('Holidays & Events Count per Year (2012-2018)')
plt.xlabel('Year')
plt.ylabel('Count of Holidays/Events')
plt.grid(False)
plt.tight_layout()
plt.legend(title='Holiday_Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# %% [markdown]
# #### Yearly Trends:
# 
# The plot shows how the number of holidays and events has changed over the years from 2012 to 2017. By examining the heights of the bars, we can identify which years had more or fewer holidays/events.
# ** **
# Holiday_Type Differences:
# 
# The different colors in the bars represent different locales. This allows us to see how the distribution of holidays/events varies across different regions or locations.
# ** **
# Business Implications:
# 
# Understanding the distribution of holidays and events can help in planning marketing campaigns, inventory management, and staffing. For instance, if certain years or Holiday_Types have consistently high numbers, businesses can prepare for increased demand during those periods.
# This visualisation provides valuable insights into the distribution of holidays and events, which can inform strategic decision-making for business operations and marketing efforts.
# 
# 

# %% [markdown]
# ### Insights pertaining to stores

# %%
# The number of unique stores in train_3

train_3['store_nbr'].nunique()

# %%
# The number of cities the stores are located 

train_3['city'].nunique()

# %%
# The number of unique states the stores are located

train_3['state'].nunique()

# %%
# Count the unique number of stores in each state
store_count = train_3.groupby('state').store_nbr.nunique().sort_values(ascending=False).reset_index()

# Plotting a horizontal bar chart to visualize the number of stores per state
plt.figure(figsize=(10, 8))
bars = plt.barh(store_count['state'], store_count['store_nbr'], color=sns.color_palette('YlOrRd', n_colors=len(store_count)))

# Adding data labels to each bar
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', 
             va='center', ha='left', fontsize=12, color='black')

max_stores = store_count['store_nbr'].max()
plt.xticks(list(range(0, max_stores + 1)))
plt.title('Number of Stores per State')
plt.xlabel('Number of Stores')
plt.ylabel('State')
plt.grid(False)  # Turn off grid
plt.tight_layout()
plt.show()

# Display store count per state
store_count


# %%
# Count the unique number of stores in each city with state as hue
store_count = train_3.groupby(['city', 'state']).store_nbr.nunique().sort_values(ascending=False).reset_index()

# Generate random colors for each unique state
unique_states = store_count['state'].unique()
random_colors = {state: f'#{random.randint(0, 0xFFFFFF):06x}' for state in unique_states}

# Plotting the unique number of stores for each city with hue as 'state'
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=store_count, y='city', x='store_nbr', hue='state', palette=random_colors, dodge=False, orient='h')

# Adding data labels to each bar
for p in ax.patches:
    if p.get_width() > 0:
        ax.annotate(format(p.get_width(), '.0f'), 
                    (p.get_width(), p.get_y() + p.get_height() / 2.), 
                    ha='center', va='center', 
                    xytext=(10, 0), 
                    textcoords='offset points',
                    color='white')

# Setting whole number units for x-axis ticks
max_stores = store_count['store_nbr'].max()
plt.xticks(list(range(0, max_stores + 1)))

plt.title('Number of Stores per City')
plt.xlabel('Number of Stores')
plt.ylabel('City')

# Move legend to the top right corner
plt.legend(title='State', bbox_to_anchor=(1.1, 1.05), loc='upper right')

plt.grid(False)  # Turn off grid lines
plt.tight_layout()
plt.show()

# Display store count per city
store_count


# %%
# Merge the train data (train_4_copy) with the stores data (train_3) based on 'store_nbr' column
merge_1 = pd.merge(train_4_copy, train_3 , on='store_nbr', how='left')

# Display a info of the dataframe
merge_1.info()


# Do a dot head to varify the merge
merge_1.head()

# %%
# Merge merge_1 with transactions data (train_5) based on 'date' and 'store_nbr' columns
merge_2 = merge_1.merge(train_5, on=['date', 'store_nbr'], how='inner')

# Check for duplicate dates in merged_df2
count_duplicate = merge_2.duplicated(subset='date', keep=False).sum()

# Display the duplicate dates
print(f"Number of duplicated dates is: {count_duplicate}\n")

# Display a info of the dataframe
merge_2.info()

# Display the first 5 rows of the merged dataframe to verify the merge operation
merge_2.head()

# %%
# Merge merge_2 with oil data based on 'date' column
merge_3 = merge_2.merge(train_1, on='date', how='left')

# Check for duplicate dates in merge_3
count_duplicate = merge_3.duplicated(subset='date', keep=False).sum()

# show duplicate dates
print(f"Number of duplicated dates is: {count_duplicate}\n")

# Display a info of the dataframe
merge_3.info()

# Display the first 5 rows of the merged dataframe to verify the merge operation
merge_3.head()

# %%
train_2.columns

# %%
# Merge merge_3 with the holidays data based on 'date', 'type', and 'year' columns
df_train = merge_3.merge(train_2, on=['date', 'year'], how='left')

# Check for duplicate dates in df_train
count_duplicate = df_train.duplicated(subset='date', keep=False).sum()

# Display the duplicate dates
print(f"Number of duplicated dates is: {count_duplicate}\n")

# Display a info of the dataframe
df_train.info()

# Display the first 5 rows of the merged dataframe to verify the merge operation
df_train.head()

# %% [markdown]
# ### **NB** 
# ** **
# merge_1 contains

# %% [markdown]
# ### **Deductions From Merging The Data And Why We Merged It The Way We Did**
# ** **
# We combined our datasets using one inner join and three left joins to create a unified dataset that includes all relevant dates for store sales, transactions, and associated data from the oil and holiday datasets. This approach ensures we maintain consistency across the dates and have all necessary information for a cleaner and more coherent time series analysis.
# 
# 
# Unique timestamps are necessary for several time series forecasting models, including LSTM, ARIMA, and Prophet. These models can be upset by duplicate timestamps, which could result in mistakes or inaccurate projections. Furthermore, when numerous observations have the same timestamp, plots and charts can become cluttered or misleading, which makes it more difficult to discern patterns and trends.
# 
# Numerical features should be adequately aggregated (using sums for sales and means or sums for promotions, depending on the situation) in order to prepare data for time series forecasting. When aggregating categorical features, such family or state, the mode should be used. This aggregated data can be effectively summarized using the groupby function, which keeps all pertinent information intact for precise modeling. Appropriate handling of missing values is also essential to guaranteeing the completeness and quality of the dataset.
# 

# %%
def duplicate_percent(df, name="DataFrame"):
    """
    Analyze the missing values and duplicated rows in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    name (str): The name of the DataFrame for display purposes.

    Returns:
    pd.DataFrame: A DataFrame containing duplicated rows if any.
    """
    # Calculate the percentage of missing values
    missing_percent = (df.isnull().mean() * 100).round(2)

    # Look for duplicated values
    count_duplicate = df.duplicated(subset=None, keep=False).sum()

    # If duplicate rows exist, show them
    duplicate_rows = df[df.duplicated(subset=None, keep=False)]

    # Show/Print Results
    print(f"Proportion of missing values in {name} dataset:")
    print(missing_percent)
    print(f"\nThe shape of {name} dataset: {df.shape}")
    print(f"\nNumber of duplicated rows in {name} dataset: {count_duplicate}")
    if count_duplicate > 0:
        print(f"\nDuplicated rows in {name} dataset:")
        print(duplicate_rows)

    # Return duplicate rows
    return df

# Check for duplicate
duplicate_rows = duplicate_percent(df_train, "df_train")


# %% [markdown]
#  `It is evident that the combined data contains null values. ` 
# 
#  `There are 21 features and 2805231 observations in the combined dataset. `
#  
#  `The combination has resulted in the renaming of four columns type_x, type_y`
# 

# %%
# Changing the column names to more appropriate ones
df_train  = df_train.rename(columns={"type_x": "store_type", "type_y": "holiday_type"})
df_train .head()

# %% [markdown]
# ** **
# 
# ** **

# %%
# Provide a synopsis of the data. 
numeric_columns = ['sales','onpromotion','cluster','transactions','dcoilwtico']
df_train[numeric_columns].describe().T

# %%
# For the column labeled "dcoilwtico," make a box plot.
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_train, y='dcoilwtico')

# Include labels and a title.
plt.title('Box Plot of Oil Prices')
plt.ylabel('Oil Prices')

# Show plot
plt.tight_layout()
plt.show()

# Show the oil price summary information.
df_train['dcoilwtico'].describe()

# %% [markdown]
# ** ** 
# ### Mean imputation would be adequate for managing missing numerical values because oil prices don't contain any outliers.
# ** **

# %%
# Calculate the number of duplicate rows in the dataframe 'df_train'
count_duplicate = df_train.duplicated().sum()

# Display the count of duplicate rows
count_duplicate 

# %%
def impute_missing_values(df, column_name, strategy='mean'):
    """
    Impute missing values in a specified column of a DataFrame using SimpleImputer.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to impute.
    strategy (str): The imputation strategy. Options are 'mean', 'median', 'most_frequent', or 'constant'.
                    Default is 'mean'.

    Returns:
    pd.DataFrame: A DataFrame with imputed values in the specified column.
    """
    # Create an instance of SimpleImputer with the specified strategy
    imputer = SimpleImputer(strategy=strategy)

    # Reshape the data to fit the imputer
    data_reshaped = df[[column_name]].values

    # Impute missing values in the specified column
    df[column_name] = imputer.fit_transform(data_reshaped)

    return df

# Check missing values
df_train = impute_missing_values(df_train, 'dcoilwtico', strategy='mean')


# %%
# Check for any remaining missing values in the 'dcoilwtico' column of df_train
# The isna() method returns a boolean DataFrame where True indicates missing values
# The sum() method counts the number of True values, effectively counting the missing values
missing_values_count = df_train['dcoilwtico'].isna().sum()

# Display the count of missing values in the 'dcoilwtico' column
missing_values_count

# %%
# Get the unique values in the 'transferred' column of df_train
# The unique() method returns an array of unique values present in the 'transferred' column
# Display the unique values in the 'transferred' column
df_train['transferred'].unique()

# %%
# Define a function to standardize values in a column
def standardize_value(value):
    # If the value is True, return 'Yes'
    if value is True:
        return 'Yes'
    # If the value is False, return 'No'
    elif value is False:
        return 'No'
    # If the value is NaN, return 'Not Applicable'
    elif pd.isna(value):
        return 'Not Applicable'
    # Otherwise, return the original value
    else:
        return value

# List of columns to standardize
columns_to_standardize = [
    'holiday_type', 'locale', 'locale_name', 'description', 'transferred'
]

# Apply the standardize_value function to each column in the columns_to_standardize list
for column in columns_to_standardize:
    df_train[column] = df_train[column].apply(standardize_value)

# Print the value counts for each standardized column to verify the changes
for column in columns_to_standardize:
    # Display the count of unique values in the column
    print(df_train[column].value_counts())
    # Print an empty line for better readability between the outputs of different columns
    print()


# %%
def count_duplicate_rows(df):
    """
    Analyze a DataFrame to calculate missing values percentage, count duplicates, and extract duplicate rows.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    dict: A dictionary containing the missing percentage, count of duplicates, and duplicate rows.
    """
    # Calculate the percentage of missing values in each column
    missing_percentage = (df.isnull().mean() * 100).round(2)

    # Count the number of duplicate rows
    count_duplicate = df.duplicated(subset=None, keep=False).sum()

    # Extract the duplicate rows
    duplicated_rows = df[df.duplicated(subset=None, keep=False)]

    # Print results
    print("Proportion of missing values in the dataset:")
    print(missing_percentage)
    print("\nThe shape of the dataset:", df.shape)
    print("\nNumber of duplicated rows in the dataset:", count_duplicate)

    # Return the results as a dictionary
    return df  #{
    #     'missing_percentage': missing_percentage,
    #     'count_duplicate': count_duplicate,
    #     'duplicated_rows': duplicated_rows
    # }

# Check for duplicates
results = count_duplicate_rows(df_train)


# %%

# Identify columns with object data type
# select_dtypes(include=['object']) filters columns with object data type
# .columns returns the column names, and the list comprehension creates a list of these column names
object_columns = [col for col in df_train.select_dtypes(include=['object']).columns]

# Convert the identified object columns to category data type
# This can help reduce memory usage as category type is more memory efficient for columns with repetitive string values
df_train[object_columns] = df_train[object_columns].astype('category')

# Display the memory usage information of the DataFrame
# memory_usage='deep' provides a detailed memory usage report, including object-type columns
df_train.info(memory_usage='deep')


# %%

def num_convert(df):
    # Select columns with float64 data type and convert them to float32
    float64_cols = df.select_dtypes(include=['float64'])
    df[float64_cols.columns] = float64_cols.astype('float32')

    # Select columns with float data type and downcast them to the smallest float subtype
    float_cols = df.select_dtypes(include=['float'])
    df[float_cols.columns] = float_cols.apply(pd.to_numeric, downcast='float')

    # Select columns with integer data type and downcast them to the smallest integer subtype
    int_cols = df.select_dtypes(include=['int'])
    df[int_cols.columns] = int_cols.apply(pd.to_numeric, downcast='integer')

    return df

# Apply the downcast_numeric function to the df_train DataFrame to optimize memory usage
df_train = num_convert(df_train)

# Display the memory usage information of the DataFrame
# memory_usage='deep' provides a detailed memory usage report, including object-type columns
df_train.info(memory_usage='deep')


# %%
# Display the unique values present in the 'day' column of the df_train DataFrame
# This provides insight into the different values or categories that are contained within the 'day' column
df_train['day'].unique()

# %%
# Identify columns with the 'category' data type in the df_train DataFrame
categorical_columns = df_train.select_dtypes('category').columns

# Display the names of the categorical columns
# This provides insight into which columns have been converted to or are stored as categorical data types
categorical_columns

# %% [markdown]
# ## `Univariate Analysis`
# Here is the section to explore, analyze, visualize each variable independently of the others.

# %%
# find unique values in each column of the dataset

for column in df_train.columns:
    print(f'{column}: {df_train[column].nunique()} unique values')


# %%
# Descriptive statistics for  sales
pd.DataFrame(df_train['sales'].describe())

# %%
# Descriptive statistics on the family column 
pd.DataFrame(df_train['family'].value_counts())

# %%
#  plot box plots for sales column

df_train.boxplot(column='sales', figsize=(10,5))

# %%
# plot KDE plot for the sales column

sns.kdeplot(df_train['sales'], shade=True)

# %%
# Visualize KDE plot for dcoilwtico

sns.kdeplot(df_train['dcoilwtico'], shade=True)

# %%
# plot KDE plot for store_nbr 

sns.kdeplot(df_train['store_nbr'], shade=True)


# %%
# Visualize cluster

sns.countplot(x='cluster', data=df_train)

# %%
#  view cluster as KDE plot

sns.kdeplot(df_train['cluster'], shade=True)

# %%
# Use KDE plot to visualize id

sns.kdeplot(df_train['id'], shade=True)

# %%
# visualize onpromotion with KDE plot

sns.kdeplot(df_train['onpromotion'], shade=True)


# %% [markdown]
# ## `Bivariate & Multivariate Analysis`
# Here is the section to explore, analyze, visualize each variable in relation to the others.

# %%
# Describe the numerical values of the dataset 

df_train.describe()

# %%
# Plotting histograms on the columns

df_train.hist(bins=30, figsize=(20,15))

# %%
# Determine the average sales over various time periods.
# Calculate average daily sales
daily_sales = train_4_copy.groupby('day')['sales'].mean().reset_index()

# Calculate average weekly sales
weekly_sales = train_4_copy.groupby('week')['sales'].mean().reset_index()

# Calculate average monthly sales
monthly_sales = train_4_copy.groupby('month')['sales'].mean().reset_index()

# Calculate average yearly sales
annualy_sales = train_4_copy.groupby('year')['sales'].mean().reset_index()


# %%
# Map the day names to ensure the week starts from Monday
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_sales['day'] = pd.Categorical(daily_sales['day'], categories=days_order, ordered=True)

# Sort the data by day names to ensure the correct order in the plot
daily_sales = daily_sales.sort_values('day')

# %%
# Create a 2x2 grid of subplots with a size of 15x10 inches
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot daily sales data on the top-left subplot
axs[0, 0].plot(daily_sales['day'], daily_sales['sales'], color='blue')
axs[0, 0].set_title('Daily Sales')  # Set title for daily sales plot
axs[0, 0].set_xlabel('Day')        # Label x-axis as 'Day'
axs[0, 0].set_ylabel('Mean Sales')  # Label y-axis as 'Mean Sales'

# Plot weekly sales data on the top-right subplot
axs[0, 1].plot(weekly_sales['week'], weekly_sales['sales'], color='green')
axs[0, 1].set_title('Weekly Sales')  # Set title for weekly sales plot
axs[0, 1].set_xlabel('Week')         # Label x-axis as 'Week'
axs[0, 1].set_ylabel('Mean Sales')   # Label y-axis as 'Mean Sales'

# Plot monthly sales data on the bottom-left subplot
axs[1, 0].plot(monthly_sales['month'], monthly_sales['sales'], color='red')
axs[1, 0].set_title('Monthly Sales')  # Set title for monthly sales plot
axs[1, 0].set_xlabel('Month')         # Label x-axis as 'Month'
axs[1, 0].set_ylabel('Mean Sales')    # Label y-axis as 'Mean Sales'

# Plot annual sales data on the bottom-right subplot
axs[1, 1].plot(annualy_sales['year'], annualy_sales['sales'], color='purple')
axs[1, 1].set_title('Annual Sales')  # Set title for annual sales plot
axs[1, 1].set_xlabel('Year')         # Label x-axis as 'Year'
axs[1, 1].set_ylabel('Mean Sales')   # Label y-axis as 'Mean Sales'
axs[1, 1].set_xticks(annualy_sales['year'].astype(int))  # Set x-ticks for years

# Adjust layout to prevent overlap of subplots
plt.tight_layout()

# Display the plots
plt.show()


# %% [markdown]
# ## Feature Processing & Engineering
# Here is the section to **clean**, **process** the dataset and **create new features**.

# %% [markdown]
# `Impute Missing Values`

# %%
df_train.isnull().sum()

# %% [markdown]
# ## Answering Hypothesis Questions

# %%
# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_train, x='date', y='sales', marker='o')
plt.title('Sales over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(df_train['date'].astype('int64') // 10**9, df_train['sales'])

# Plot the regression line
plt.plot(df_train['date'], intercept + slope * (df_train['date'].astype('int64') // 10**9), color='red', lw=2, label=f'Linear fit: y={intercept:.2f} + {slope:.2f}x')
plt.legend()

# Display the plot
plt.show()

# Summary of linear regression
{
    'slope': slope,
    'intercept': intercept,
    'r_value': r_value,
    'p_value': p_value,
    'std_err': std_err
}


# %% [markdown]
# ### Deduction
# ** **
# `Null Hypothesis (H0):`
# There is no significant relationship between the date and the sales figures in the dataset.
# 
# `Alternative Hypothesis (H1):`
# There is a significant relationship between the date and the sales figures in the dataset.
# 
# `Linear Regression Analysis:`
# The key statistics from the regression analysis are as follows:
# 
# **Slope: 1.96e-06**
# 
# **Intercept: -2412.31**
# 
# **R-squared value (R²): 0.0717**
# 
# **P-value: 0.0**
# 
# **Standard Error: 1.625e-08**
# 
# 
# `Interpretation:`
# Slope: The slope of the regression line is 1.96e-06, indicating a very slight positive relationship between date and sales. This means that as time progresses, sales figures show a minor increase.
# 
# `Intercept:` The intercept is -2412.31, which indicates the starting point of the regression line when the date is zero. This value is more abstract in terms of direct interpretation but is part of the regression equation.
# 
# `R-squared value (R²):` The R² value is 0.0717, suggesting that the model explains only about 7.17% of the variability in sales data. While this is a low value, it indicates some degree of relationship.
# 
# `P-value:` The p-value is 0.0, which is significantly lower than the common significance level of 0.05. This indicates strong evidence against the null hypothesis, suggesting that the relationship between date and sales is statistically significant.
# 
# `Standard Error:` The standard error of 1.625e-08 indicates the precision of the estimated slope. A very small standard error suggests that the estimate is precise.
# 
# ### Conclusion: 
# Based on the linear regression analysis, there is a statistically significant relationship between the date and the sales figures in the dataset. The low p-value leads us to reject the null hypothesis in favor of the alternative hypothesis, indicating a significant relationship.
# 
# While the R² value is relatively low, it does show that the date can explain a small part of the variability in sales figures. Thus, despite the weak positive relationship, the time element does play a significant role in influencing sales figures in this dataset.
# 
# 
# **we do not accept the null hypothesis.**
# ** **

# %% [markdown]
# ## Answering Analytical Questions

# %%
# 1. Is the train dataset complete (has all the required dates)?
# We'll create a date range from the minimum to the maximum date in the dataset
date_range = pd.date_range(start=train_4_copy['date'].min(), end=train_4_copy['date'].max())

# Check for any missing dates in the dataset
dates_missing = date_range.difference(train_4_copy['date'])
is_complete = dates_missing.empty

is_complete, dates_missing

# %%
# 2. Which dates have the lowest and highest sales for each year (excluding days the store was closed)?
# Filter the dataframe to include only rows where sales are greater than zero
df_nonzero_sales = df_train[df_train['sales'] != 0]

# Identify the dates with the highest sales for each year and select relevant columns
highest_sales_dates = df_nonzero_sales.loc[df_nonzero_sales.groupby('year')['sales'].idxmax()][['year', 'date', 'sales']]

# Identify the dates with the lowest sales for each year and select relevant columns
lowest_sales_dates = df_nonzero_sales.loc[df_nonzero_sales.groupby('year')['sales'].idxmin()][['year', 'date', 'sales']]

# Create a figure and two subplots for highest and lowest sales
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Plot the highest sales dates
sns.barplot(data=highest_sales_dates, x='date', y='sales', ax=ax[0], palette='magma')
ax[0].set_title('Highest Sales Dates for Each Year')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Sales')
ax[0].set_xticklabels(highest_sales_dates['year'].astype(str), rotation=45)

# Plot the lowest sales dates
sns.barplot(data=lowest_sales_dates, x='date', y='sales', ax=ax[1], palette='magma')
ax[1].set_title('Lowest Sales Dates for Each Year')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Sales')
ax[1].set_xticklabels(lowest_sales_dates['year'].astype(str), rotation=45)

plt.tight_layout()
plt.show()

# %%
# 3. Compare the sales for each month across the years and determine which month of which year had the highest sales.

# Group by 'year' and 'month' and sum the sales
high_yr_mth_sales = train_4_copy.groupby(['year', 'month'])['sales'].sum().reset_index()

# Get the month with the highest sales for each year
highest_sales_per_year = high_yr_mth_sales.loc[high_yr_mth_sales.groupby('year')['sales'].idxmax()]

# Convert numerical month to text month
highest_sales_per_year['month'] = highest_sales_per_year['month'].apply(lambda x: calendar.month_name[x])

# Combine 'year' and 'month' into a single column for plotting
highest_sales_per_year['year_month'] = highest_sales_per_year['year'].astype(str) + '-' + highest_sales_per_year['month']

# Plotting the bar plot
plt.figure(figsize=(12, 8))
sns.barplot(data=highest_sales_per_year, x='year_month', y='sales', palette='viridis')

# Adding labels and title
plt.xlabel('Month-Year')
plt.ylabel('Sales')
plt.title('Highest Sales Month for Each Year')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()



# %%
# 4. Did the earthquake impact sales?
# Filter data around the earthquake date to observe changes in sales
# Define the earthquake date
earthquake_date = pd.Timestamp('2016-04-16')

# Filter data for the months before and after the earthquake
earthquake_impact_window = df_train[(df_train['date'] >= '2016-03-01') & (df_train['date'] <= '2016-06-30')]

# Group by date and sum sales to observe changes
earthquake_sales_trends = earthquake_impact_window.groupby('date')['sales'].sum().reset_index()

# Plotting to visualize sales trends around the earthquake
plt.figure(figsize=(14, 7))
sns.lineplot(x='date', y='sales', data=earthquake_sales_trends, marker='o')
plt.title('Sales Trends Around the Earthquake Date (March - June 2016)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.axvline(x=earthquake_date, color='red', linestyle='--', label='Earthquake Date')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# #### Observations from the Graph:
# - Pre-Earthquake Sales: Prior to the earthquake, sales show a general trend of oscillation with no significant peaks. This is typical for regular business operations without extraordinary external influences.
# - Post-Earthquake Sales: Following the earthquake, there appears to be an increase in sales volume on certain dates, suggesting that the earthquake might have influenced buying patterns. Peaks in sales could potentially indicate a surge in purchases of essentials due to relief efforts and personal preparedness by the population.
# ** **

# %%
df_train.head(1)

# %%
# 5 Are certain stores or groups of stores selling more products? (Cluster, city, state, type)
# Analyze sales based on different categories such as cluster, city, state, and store type

# Summing sales for each cluster
sales_by_cluster = df_train.groupby('cluster')['sales'].sum().sort_values(ascending=False)

# Summing sales for each city
sales_by_city = df_train.groupby('city')['sales'].sum().sort_values(ascending=False)

# Summing sales for each state
sales_by_state = df_train.groupby('state')['sales'].sum().sort_values(ascending=False)

# Summing sales for each store type
sales_by_store_type = df_train.groupby('store_type')['sales'].sum().sort_values(ascending=False)

print(sales_by_cluster, sales_by_city, sales_by_state, sales_by_store_type)

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot sales by cluster
sns.barplot(x=sales_by_cluster.index, y=sales_by_cluster.values, ax=axs[0, 0], palette="viridis")
axs[0, 0].set_title('Sales by Cluster')
axs[0, 0].set_xlabel('Cluster')
axs[0, 0].set_ylabel('Total Sales')
axs[0, 0].tick_params(axis='x', rotation=45)

# Plot sales by city
sns.barplot(x=sales_by_city.index, y=sales_by_city.values, ax=axs[0, 1], palette="magma")
axs[0, 1].set_title('Sales by City')
axs[0, 1].set_xlabel('City')
axs[0, 1].set_ylabel('Total Sales')
axs[0, 1].tick_params(axis='x', rotation=90)

# Plot sales by state
sns.barplot(x=sales_by_state.index, y=sales_by_state.values, ax=axs[1, 0], palette="plasma")
axs[1, 0].set_title('Sales by State')
axs[1, 0].set_xlabel('State')
axs[1, 0].set_ylabel('Total Sales')
axs[1, 0].tick_params(axis='x', rotation=45)

# Plot sales by store type
sns.barplot(x=sales_by_store_type.index, y=sales_by_store_type.values, ax=axs[1, 1], palette="coolwarm")
axs[1, 1].set_title('Sales by Store Type')
axs[1, 1].set_xlabel('Store Type')
axs[1, 1].set_ylabel('Total Sales')
axs[1, 1].tick_params(axis='x', rotation=45)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()



# %%
# 6 Are sales affected by promotions, oil prices and holidays?

# Analyzing the effect of promotions, oil prices, and holidays on sales

# Grouping by promotions and calculating average sales
promotion_sales = df_train.groupby('onpromotion')['sales'].mean()

# Correlation between oil prices and sales
oil_price_correlation = df_train['sales'].corr(df_train['dcoilwtico'])

# Grouping by holiday status and calculating average sales
holiday_sales = df_train[df_train['holiday_type'] != 'Not Applicable'].groupby('holiday_type')['sales'].mean()


# Calculate the correlation matrix
correlation_matrix = df_train[['sales', 'dcoilwtico', 'onpromotion']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

print(promotion_sales, oil_price_correlation, holiday_sales)

# %% [markdown]
# - Promotions: If average sales are higher on promotional days, it indicates that promotions positively affect sales.
# - Oil Prices: A negative correlation between sales and oil prices would suggest that higher oil prices are associated with lower sales.
# - Holidays: Comparing sales during holidays vs. non-holidays can show whether holidays have a significant impact on sales.
# ** **

# %%
# 7 What analysis can we get from the date and its extractable features
#  Seasonal Trends
# Group by month to observe seasonal trends
seasonal_sales = df_train.groupby('month')['sales'].sum()

# Plotting seasonal sales trends
plt.figure(figsize=(12, 6))
sns.lineplot(x=seasonal_sales.index, y=seasonal_sales.values)
plt.title('Seasonal Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(False)
plt.show()


# Day of the Week Analysis
# Group by day of the week to observe trends
weekday_sales = df_train.groupby('day')['sales'].mean()

# Plotting sales by day of the week
plt.figure(figsize=(12, 6))
sns.barplot(x=weekday_sales.index, y=weekday_sales.values)
plt.title('Average Sales by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Sales')
plt.grid(False)
plt.show()

# Group by year and month and sum the sales
monthly_sales = df_train.groupby(['year', 'month'])['sales'].sum().reset_index()

# Plotting the bar plot with subplots for each year
g = sns.FacetGrid(monthly_sales, col="year", col_wrap=3, height=4, aspect=1.5)
g.map(sns.barplot, "month", "sales", palette="viridis")
g.set_axis_labels("Month", "Total Sales")
g.set_titles("{col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Monthly Sales Trends Over Years')
plt.show()


# Holiday Impact
# Group by holiday type to observe sales impact
holiday_sales = df_train.groupby('holiday_type')['sales'].mean()

# Plotting sales by holiday type
plt.figure(figsize=(12, 6))
sns.barplot(x=holiday_sales.index, y=holiday_sales.values)
plt.title('Average Sales by Holiday Type')
plt.xlabel('Holiday Type')
plt.ylabel('Average Sales')
plt.grid(False)
plt.show()

# Sales Growth Over Time
# Plotting sales over time
plt.figure(figsize=(14, 7))
sns.lineplot(x='date', y='sales', data=df_train)
plt.title('Sales Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(False)
plt.show()


# %% [markdown]
# - By extracting features such as month, day of the week, and year from the date, and analyzing these features, we can gain insights into seasonal sales patterns, the impact of specific days and holidays on sales, long-term sales trends, and overall sales growth. These analyses can help in making data-driven decisions for inventory management, promotional strategies, and resource allocation.
# ** **

# %%
# 8 Which product family and stores did the promotions affect.
# Calculate Average Sales During Promotions:
df_promo = df_train[df_train['onpromotion'] > 0]
avg_sales_promo = df_promo.groupby(['family', 'store_nbr'])['sales'].mean().reset_index()
avg_sales_promo.rename(columns={'sales': 'avg_sales_promo'}, inplace=True)
avg_sales_promo

# %%

# Calculate Average Sales Without Promotions:
df_no_promo = df_train[df_train['onpromotion'] == 0]
avg_sales_no_promo = df_no_promo.groupby(['family', 'store_nbr'])['sales'].mean().reset_index()
avg_sales_no_promo.rename(columns={'sales': 'avg_sales_no_promo'}, inplace=True)

# Merge Data:
avg_sales = pd.merge(avg_sales_promo, avg_sales_no_promo, on=['family', 'store_nbr'], how='outer')
avg_sales['sales_difference'] = avg_sales['avg_sales_promo'] - avg_sales['avg_sales_no_promo']

# Identify the Most Affected Families and Stores:
affected = avg_sales.sort_values(by='sales_difference', ascending=False)
affected

# %% [markdown]
# - Effectiveness of Promotions: Promotions can significantly boost sales, but the impact varies across different product families and stores. Tailoring promotional strategies to specific product categories and store characteristics can enhance effectiveness.
# - Oil Prices: Oil prices have an indirect effect on sales through transportation costs, affecting product categories differently.
# - Holidays: Holidays influence sales patterns, with certain product families experiencing notable increases in sales during these periods.
# ** **

# %% [markdown]
# ### 9. What is the difference between RMSLE, RMSE, MSE (or why is the MAE greater than all of them?)
# - **MSE:** Penalizes larger errors due to squaring. Sensitive to outliers.
# - **RMSE:** Square root of MSE, providing an error metric in the same units as the target.
# - **RMSLE:** Useful for targets spanning multiple orders of magnitude, less sensitive to large errors due to logarithmic scaling.
# - **MAE:** Linear error metric, less sensitive to outliers compared to MSE and RMSE.

# %%
# 10
# Reset index to access the date column
df = df_train.reset_index()

# Convert the 'date' index to a datetime object if not already
df['date'] = pd.to_datetime(df['date'])

# Identify wage payment days
df['is_wage_payment_day'] = df['date'].apply(lambda x: x.day in [15, x.days_in_month])

# Calculate average sales on wage payment days and other days
avg_sales_wage_payment = df[df['is_wage_payment_day']].groupby(['family', 'store_nbr'])['sales'].mean().reset_index()
avg_sales_wage_payment.rename(columns={'sales': 'avg_sales_wage_payment'}, inplace=True)

avg_sales_non_wage_payment = df[~df['is_wage_payment_day']].groupby(['family', 'store_nbr'])['sales'].mean().reset_index()
avg_sales_non_wage_payment.rename(columns={'sales': 'avg_sales_non_wage_payment'}, inplace=True)

# Merge Data
avg_sales = pd.merge(avg_sales_wage_payment, avg_sales_non_wage_payment, on=['family', 'store_nbr'], how='outer')
avg_sales['sales_difference'] = avg_sales['avg_sales_wage_payment'] - avg_sales['avg_sales_non_wage_payment']

# Identify the most affected families and stores
affected = avg_sales.sort_values(by='sales_difference', ascending=False)

# Plotting with violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='family', y='sales', hue='is_wage_payment_day', data=df, split=True)
plt.xticks(rotation=90)
plt.title('Sales Distribution on Wage Payment Days vs Other Days')
plt.xlabel('Product Family')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()

# Perform a t-test to compare sales on payment and non-payment days
payment_days_sales = df[df['is_wage_payment_day']]['sales']
non_payment_days_sales = df[~df['is_wage_payment_day']]['sales']

t_stat, p_value = stats.ttest_ind(payment_days_sales, non_payment_days_sales)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")



# %% [markdown]
# ## Machine Learning Processes And Models 

# %%
# # Functions used 
# convert_date_and_info
# add_date_parts
# analyze_dataframes
# duplicate_percent
# standardize_value
# count_duplicate_rows
# num_convert


# %%
# # Ensure the target column 'sales' is numeric and the index is datetime
# df_train['sales'] = pd.to_numeric(df_train['sales'], errors='coerce')
# df_train = df_train.dropna(subset=['sales'])
# df_train.set_index('date', inplace=True)
# df_train.index = pd.to_datetime(df_train.index)


# %% [markdown]
# ` Pipeline for processing `

# %%
# A combination of all the functions used in this notebook for creating a pipeline
def combined_preprocessing(df, date_column='date', impute_column='dcoilwtico', impute_strategy='mean', columns_to_standardize=None):
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Add date parts
    df['day'] = df[date_column].dt.day_name()
    df['month'] = df[date_column].dt.month
    df['week'] = df[date_column].dt.isocalendar().week.astype(int)
    df['year'] = df[date_column].dt.year

    # Analyze missing values and duplicates
    print("Proportion of missing values:")
    print((df.isnull().mean() * 100).round(2))
    print("\nNumber of duplicated rows:", df.duplicated().sum())

    # Impute missing values
    if impute_column in df.columns:
        imputer = SimpleImputer(strategy=impute_strategy)
        df[impute_column] = imputer.fit_transform(df[[impute_column]])
    else:
        print(f"Column '{impute_column}' not found in DataFrame. Skipping imputation step.")

    # Standardize certain columns
    if columns_to_standardize is not None:
        for column in columns_to_standardize:
            if column in df.columns:
                df[column] = df[column].apply(lambda x: 'Yes' if x is True else ('No' if x is False else ('Not Applicable' if pd.isna(x) else x)))
            else:
                print(f"Column '{column}' not found in DataFrame. Skipping standardization step for this column.")
    
     # Print value counts for standardized columns
    if columns_to_standardize is not None:
        for column in columns_to_standardize:
            if column in df.columns:
                print(df[column].value_counts())
                print()

    # Convert numeric columns to optimize memory usage
    float64_cols = df.select_dtypes(include=['float64'])
    df[float64_cols.columns] = float64_cols.astype('float32')
    float_cols = df.select_dtypes(include=['float'])
    df[float_cols.columns] = float_cols.apply(pd.to_numeric, downcast='float')
    int_cols = df.select_dtypes(include=['int'])
    df[int_cols.columns] = int_cols.apply(pd.to_numeric, downcast='integer')

    return df

# Define columns to standardize
columns_to_standardize = ['holiday_type', 'locale', 'locale_name', 'description', 'transferred']

# # Apply combined preprocessing function
# df_test_processed = combined_preprocessing(df_test, date_column='date', impute_column='dcoilwtico', columns_to_standardize=columns_to_standardize)
# print(df_test_processed)


# %%
# Creating the preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('combined_preprocessing', FunctionTransformer(combined_preprocessing, 
                                                   kw_args={
                                                       'date_column': 'date', 
                                                       'impute_column': 'dcoilwtico', 
                                                       'columns_to_standardize': columns_to_standardize
                                                   }))
])

# %%
#  The data has been split already, the test data is different from the train data
# Read the first test data into a dataframe
test_1 = pd.read_csv(r"C:\Users\Safowaa\Documents\Azibiafrica\AzubiPython\The_Regression_Project\Project_data\test.csv")
test_1.head()

# %%
# Read the second test data into a dataframe
test_2 = pd.read_csv(r"C:\Users\Safowaa\Documents\Azibiafrica\AzubiPython\The_Regression_Project\Project_data\sample_submission_test.csv")
test_2.head()

# %%
# combine all the test data into one for testing
df_test = pd.merge(test_1, test_2 , on='id', how='left')

# Display a info of the dataframe
df_test.info()

# %% [markdown]
# ` Tranforming the test data set `

# %%
# Apply the pipeline to the test data
preprocessing_pipeline.fit_transform(df_test)
df_test


# %% [markdown]
# `Dataset Splitting`

# %%
# Split data for machine learning models
train_ml, val_ml = train_test_split(df_train, test_size=0.2, shuffle=True)

# %%
# Define a mapping from day names to integers
day_to_int = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

# Map the day names to integers
df_train_copy1['day'] = df_train_copy1['day'].map(day_to_int)
df_test_copy1['day'] = df_test_copy1['day'].map(day_to_int)

# %%
# Create a copy of the dataframe 
df_train_copy1 = df_train.copy()   #RandomForest
df_train_copy2 = df_train.copy()   #XGBoost
df_test_copy1 = df_test.copy()     #RandonForest
df_test_copy2 = df_test.copy()     #XGBoost

# %%
# Split data for RandomForest and XGBoost
train_ml1, val_ml1 = train_test_split(df_train_copy1, test_size=0.2, shuffle=True)

# %%
# Split the data for machine learning model RandomForest

Xdf_train_copy1 = df_train_copy1.drop("sales", axis= 1)

ydf_train_copy1 = df_train_copy1["sales"]

Xdf_test_copy1 = df_test_copy1.drop("sales", axis= 1)

ydf_test_copy1 = df_test_copy1["sales"]

# %% [markdown]
# `Features Creation` **&** `Encoding `

# %% [markdown]
# 1. For Time Series Models (ARIMA, SARIMA, ETS, Prophet):
# - Time-based Features:
# - Lag Features: Create features that represent past values of the target variable (e.g., sales) at different time lags.

# %%
df_train['lag_1'] = df_train['sales'].shift(1)
df_train['lag_2'] = df_train['sales'].shift(2)


# %% [markdown]
# - Lag features are created by shifting the target variable  (sales) by a certain number of time periods (lags). This means you use past values of the variable to predict future values.
# - y including past values, the model can learn patterns such as trends and seasonality, improving its predictive accuracy.
# - Lag features allow the model to understand how past values of the series affect the current value. This is important in time series data where previous observations can have a significant impact on future outcomes.

# %% [markdown]
# - Rolling Statistics: Calculate rolling means, sums, or standard deviations over a window.

# %%
df_train['rolling_mean_7'] = df_train['sales'].rolling(window=7).mean()


# %%
df_train['rolling_mean_7']

# %% [markdown]
# - Our  7-day rolling mean calculates the average of the current day and the previous six days.
# - It smooths out short-term fluctuations and highlights longer-term trends or cycles in the data.
# - It helps in identifying trends over time, making it easier to see patterns or changes in direction.
# - Finally, helps to visualize trends in the data over a weekly period, reducing day-to-day variability.

# %% [markdown]
# - Seasonality and Trends:
# - Seasonal Indicators: For SARIMA and ETS, use indicators for seasons or holidays.

# %%
df_train['is_holiday'] = df_train['holiday_type'].apply(lambda x: 1 if x == 'Holiday' else 0)


# %%
# Check for invalid dates in the date column
def check_invalid_dates(df, date_column='date'):
    try:
        pd.to_datetime(df[date_column])
        print("All dates are valid.")
    except Exception as e:
        print(f"Invalid date found: {e}")
        
check_invalid_dates(df_train)
check_invalid_dates(df_test)


# %%

# Identify and remove rows with invalid dates
def clean_invalid_dates(df, date_column='date'):
    df = df.reset_index()  # Reset index to ensure 'date' is a column
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    invalid_dates = df[df[date_column].isna()]
    if not invalid_dates.empty:
        print("Removing rows with invalid dates:")
        print(invalid_dates)
        df = df.dropna(subset=[date_column])
    return df.set_index(date_column)  # Set 'date' column back as index

df_train = clean_invalid_dates(df_train)
df_test = clean_invalid_dates(df_test)

# Ensure the date columns are in the correct format for Prophet
df_train['ds'] = df_train.index
df_train['y'] = df_train['sales']
df_test['ds'] = df_test.index

# %% [markdown]
# - For Machine Learning Models (RandomForest, XGBoost):
# - Lag Features and Rolling Statistics: As with time series models, these are useful for capturing temporal patterns.
# 
#  Categorical Encoding:
# 
# - One-Hot Encoding: For categorical variables such as family, city, state

# %%
df_train_copy1 = pd.get_dummies(df_train_copy1, columns=['family', 'city', 'state', 'store_type', 'holiday_type', 'locale', 'locale_name', 'description', 'transferred'])

#One-hot encode categorical columns
Xdf_train_copy1_encoded = pd.get_dummies(Xdf_train_copy1, drop_first=True)
Xdf_test_copy1_encoded = pd.get_dummies(Xdf_test_copy1, drop_first=True)

#Align the train and test sets
Xdf_train_copy1_encoded, Xm_test_encoded = Xdf_train_copy1_encoded.align(Xdf_test_copy1_encoded, join='left', axis=1, fill_value=0)


# %% [markdown]
# - Product of Features: Create new features by combining existing features to capture interactions.

# %%
df_train_copy1['sales_onpromotion'] = df_train_copy1['sales'] * df_train_copy1['onpromotion']
df_test_copy1['sales_onpromotion'] = df_test_copy1['sales'] * df_test_copy1['onpromotion']

# %% [markdown]
# `Features Scaling`
# 

# %% [markdown]
# - Feature Scaling: Normalize or standardize numerical features to help improve model performance.

# %%
scaler = StandardScaler()
df_train_copy1[['sales', 'onpromotion', 'transactions', 'dcoilwtico']] = scaler.fit_transform(df_train_copy1[['sales', 'onpromotion', 'transactions', 'dcoilwtico']])

# %% [markdown]
# ## Machine Learning Modeling 
# ### Model Creation

# %%
models = {}

models['ARIMA'] = lambda df: ARIMA(df['sales'], order=(5,1,0)).fit()
models['SARIMA'] = lambda df: SARIMAX(df['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
models['ETS'] = lambda df: ExponentialSmoothing(df['sales'], seasonal='add', seasonal_periods=12).fit()
models['Prophet'] = lambda df: Prophet().fit(pd.DataFrame({'ds': df.index, 'y': df['sales']}))
# models['RandomForest'] = lambda df: RandomForestRegressor(n_estimators=100).fit(df.drop('sales', axis=1), df['sales'])
# models['XGBoost'] = lambda df: xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100).fit(df.drop('sales', axis=1), df['sales'])


# %%
def train_and_predict(model_name, df_train, val_data, df_test):
    # Ensure the model name exists in the dictionary
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not defined in the models dictionary.")
    
    # Initialize the model
    model = models[model_name](df_train)

    # Check model type and make predictions
    if model_name in ['ARIMA', 'SARIMA', 'ETS']:
        val_predictions = model.forecast(len(val_data))
        test_predictions = model.forecast(len(df_test))
    elif model_name == 'Prophet':
        future_val = pd.DataFrame({'ds': val_data.index})
        future_test = pd.DataFrame({'ds': df_test.index})
        val_predictions = model.predict(future_val)['yhat']
        test_predictions = model.predict(future_test)['yhat']
    else:  # For machine learning models
        val_predictions = model.predict(val_data.drop('sales', axis=1))
        test_predictions = model.predict(df_test.drop('sales', axis=1))
    
    return val_predictions, test_predictions


# %% [markdown]
# ## Model **1)**   ARIMA
# 
# 

# %% [markdown]
# ### Call the Model

# %%
model_name = 'ARIMA' 

# %% [markdown]
# ### Evaluate Model on Validation and testing data set

# %%
val_predictions, test_predictions = train_and_predict(model_name, df_train, val_ml, df_test)

# %%
# The MemoryError indicates that your data is too large to fit into memory when using certain models, particularly the ARIMA model. Here's a more efficient approach to handle this:

# %% [markdown]
# ### View Results

# %%
print("Validation Predictions:", val_predictions)
print("Test Predictions:", test_predictions)

# %%
# # Validation Predictions: 2805231    394.162823
# 2805232    745.195494
# 2805233    787.659829
# 2805234    722.388079
# 2805235    354.675957
#               ...    
# 3366273    576.338300
# 3366274    576.338300
# 3366275    576.338300
# 3366276    576.338300
# 3366277    576.338300
# Name: predicted_mean, Length: 561047, dtype: float64
# Test Predictions: 2805231    394.162823
# 2805232    745.195494
# 2805233    787.659829
# 2805234    722.388079
# 2805235    354.675957
#               ...    
# 2833738    576.338300
# 2833739    576.338300
# 2833740    576.338300
# 2833741    576.338300
# 2833742    576.338300
# Name: predicted_mean, Length: 28512, dtype: float64


# %% [markdown]
# ## Model **2)** SARIMAX

# %% [markdown]
# ### Call the Model

# %%
# model_name = 'SARIMA'

# %% [markdown]
# ### Evaluate Model on Validation and testing data set

# %%
# val_predictions, test_predictions = train_and_predict(model_name, df_train, val_ml, df_test)

# %% [markdown]
# ### View Results

# %%
# print("Validation Predictions:", val_predictions)
# print("Test Predictions:", test_predictions)

# %% [markdown]
# ## Model **3)** ETS

# %% [markdown]
# ### Call the Model

# %%
model_name = 'ETS'

# %% [markdown]
# ### Evaluate Model on Validation and testing data set

# %%
val_predictions, test_predictions = train_and_predict(model_name, df_train, val_ml, df_test)

# %% [markdown]
# ### View Results

# %%
print("Validation Predictions:", val_predictions)
print("Test Predictions:", test_predictions)

# %% [markdown]
# ## Model **4)** Prophet

# %% [markdown]
# ### Call the model

# %%
model_name ='Prophet'

# %% [markdown]
# ### Evaluate Model on Validation and testing data set

# %%
val_predictions, test_predictions = train_and_predict(model_name, df_train, val_ml, df_test)

# %% [markdown]
# ### View Results

# %%
print("Validation Predictions:", val_predictions)
print("Test Predictions:", test_predictions)

# %% [markdown]
# ## Model **5)** RandomForest

# %%
# # Clean invalid dates in both training and test data
# def clean_invalid_dates(df, date_column='date'):
#     if date_column not in df.index.names:
#         raise KeyError(f"'{date_column}' column is not present as an index in the DataFrame.")
#     df.reset_index(inplace=True)  # Temporarily reset index to handle 'date' as a column
#     df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
#     invalid_dates = df[df[date_column].isna()]
#     if not invalid_dates.empty:
#         print("Removing rows with invalid dates:")
#         print(invalid_dates)
#     df = df.dropna(subset=[date_column])
#     df.set_index(date_column, inplace=True)  # Set 'date' back as index
#     return df


# %%

# # Apply the function on the train and test data
# df_train = clean_invalid_dates(df_train)
# df_test = clean_invalid_dates(df_test)


# %%

# Ensure the target column 'sales' is numeric
df_train['sales'] = pd.to_numeric(df_train['sales'], errors='coerce')
df_train = df_train.dropna(subset=['sales'])


# %%

# # Create a copy of the dataframe
# df_train_copy1 = df_train.copy()


# %%

# Add missing columns in the test dataset and fill them with zeros
missing_cols = set(df_train_copy1.columns) - set(df_test_copy1.columns)
for col in missing_cols:
    df_test_copy1[col] = 0


# %%

# Ensure consistent columns in train and test sets
df_train_copy1, df_test_copy1 = df_train_copy1.align(df_test, join='inner', axis=1, fill_value=0)


# %%
# Encode categorical features
# df_train_copy1 = pd.get_dummies(df_train_copy1)
# df_test = pd.get_dummies(df_test)

# %%
# Convert categorical features to numerical using one-hot encoding
# df_train_encoded = pd.get_dummies(df_train_copy1)
# df_test_encoded = pd.get_dummies(df_test)

# Align the columns of train and test sets
# df_train_encoded, df_test_encoded = df_train_encoded.align(df_test_encoded, join='left', axis=1, fill_value=0)

# %%
# df_train_copy1 = df_train_encoded
# df_test = df_test_encoded

# %%

# Additional Feature Engineering
# df_train_copy1['sales_onpromotion'] = df_train['sales'] * df_train['onpromotion']
# df_test['sales_onpromotion'] = df_test['sales'] * df_test['onpromotion']


# %%

# Feature Scaling
# scaler = StandardScaler()
# numeric_cols = ['sales', 'onpromotion', 'transactions', 'dcoilwtico']
# df_train_copy1[numeric_cols] = scaler.fit_transform(df_train_copy1[numeric_cols])
# df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])


# %%

# # Train and Predict Function
# def train_and_predict(model_name, df_train_copy1, val_data, df_test_copy1):
#     if model_name not in models:
#         raise ValueError(f"Model '{model_name}' is not defined in the models dictionary.")
#     model = models[model_name](df_train_copy1)

#     val_predictions = model.predict(val_data.drop('sales', axis=1))
#     test_predictions = model.predict(df_test_copy1.drop('sales', axis=1))
    
#     return val_predictions, test_predictions


# %%
# Drop datetime columns from the training data
# df_train_copy1 = df_train_copy1.select_dtypes(exclude=['datetime', 'datetime64[ns]'])
# df_test_copy1 = df_test_copy1.select_dtypes(exclude=['datetime', 'datetime64[ns]'])

# Ensure the target column 'sales' is numeric
# df_train_copy1['sales'] = pd.to_numeric(df_train_copy1['sales'], errors='coerce')

# %%
def batch_train_predict(model, df_train_copy1, val_data, df_test_copy1, batch_size=10000):
    # Shuffle the training data
    df_train_copy1 = shuffle(df_train_copy1)

    # Create empty lists to hold predictions
    val_predictions = []
    test_predictions = []

    # Determine the number of batches
    num_batches = int(np.ceil(df_train_copy1.shape[0] / batch_size))

    # Train in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, df_train_copy1.shape[0])
        
        batch = df_train_copy1.iloc[start_idx:end_idx]
        
        # Ensure the batch has no datetime columns
        batch = batch.select_dtypes(exclude=['datetime64'])
        
        # Fit the model on the batch
        model.fit(batch.drop('sales', axis=1), batch['sales'])

        # Predict on validation and test data
        val_pred_batch = model.predict(val_data.drop('sales', axis=1))
        test_pred_batch = model.predict(df_test_copy1.drop('sales', axis=1))

        val_predictions.extend(val_pred_batch)
        test_predictions.extend(test_pred_batch)

    return np.array(val_predictions), np.array(test_predictions)


# %%

# Define models with batch processing
models = {}

models['RandomForest'] = lambda df: batch_train_predict(RandomForestRegressor(n_estimators=100), df_train_copy1, val_ml1, df_test_copy1, batch_size=10000)
models['XGBoost'] = lambda df: batch_train_predict(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100), df, val_ml1, df_test_copy1, batch_size=10000)


# %%

# def train_and_predict(model_name, df_train, val_data, df_test):
#     if model_name not in models:
#         raise ValueError(f"Model '{model_name}' is not defined in the models dictionary.")
#     val_predictions, test_predictions = models[model_name](df_train)
#     return val_predictions, test_predictions


# %% [markdown]
# ### Call the model

# %%

# Train and evaluate models
model_name = 'RandomForest'


# %%
# df_train_copy1.dtypes

# %%
# df_train_copy1.info()

# %%
df_test_copy1.head(1)

# %%
df_train_copy1.head(1)

# %%
pd.get_dummies(df_test_copy1, columns=['day'])
# pd.get_dummies(df_train_copy1, columns=['day'])


# %%
val_predictions, test_predictions = train_and_predict(model_name, df_train_copy1, val_ml1, df_test_copy1)


# %%
# # Create a boolean DataFrame indicating where 'AUTOMOTIVE' is found
# mask = df_train_copy1.applymap(lambda x: 'AUTOMOTIVE' in str(x).upper())

# # Get the rows where 'AUTOMOTIVE' is found in any column
# df_train_copy1[mask.any(axis=1)]

# %%
# # Create a boolean DataFrame indicating where 'AUTOMOTIVE' is found
# mask = df_test.applymap(lambda x: 'AUTOMOTIVE' in str(x).upper())

# # Get the rows where 'AUTOMOTIVE' is found in any column
# df_test[mask.any(axis=1)]

# %%
# df_test.sample(15)

# when i took a secound look at the df_train data if had dcoilwtico, lag_1, lag_2, rolling_mean_7, and  y as float and ds as a date column aside date being the index.
# For the df_test data it has sales, onpromotion, transactions, dcoilwtico, y, and sales_onpromotion as float and ds as a date column aside date being the index.

# %% [markdown]
# ### View Results

# %%
print("Validation Predictions:", val_predictions)
print("Test Predictions:", test_predictions)

# %% [markdown]
# ## Model **6)** XGBoost

# %% [markdown]
# ### Call the model

# %%
model_name ='XGBoost'

# %% [markdown]
# ### Evaluate Model on Validation and testing data set

# %%
val_predictions, test_predictions = train_and_predict(model_name, df_train_copy1, val_ml1, df_test_copy1)

# %% [markdown]
# ### View Results

# %%
print("Validation Predictions:", val_predictions)
print("Test Predictions:", test_predictions)

# %% [markdown]
# ## Models comparison
# 

# %%
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# %%
results = []

for model_name in models.keys():
    try:
        val_predictions, test_predictions = train_and_predict(model_name, df_train, val_ml, df_test)
        mae, mse, rmse = evaluate_model(val_ml['sales'], val_predictions)
        results.append({
            'Model_Name': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Details': 'Validation Set'
        })
    except Exception as e:
        print(f"Error with model {model_name}: {e}")

# %%
# Convert results to DataFrame
results_df = pd.DataFrame(results)

# %%
# Sort the DataFrame by MAE (or any other metric)
results_df = results_df.sort_values(by='MAE')
print(results_df)

# %% [markdown]
# Create a pandas dataframe that will allow you to compare your models.
# 
# 
# |     | Model_Name     | Metric (metric_name)    | Details  |
# |:---:|:--------------:|:--------------:|:-----------------:|
# | 0   |  -             |  -             | -                 |
# | 1   |  -             |  -             | -                 |
# 
# 
# You might use the pandas dataframe method `.sort_values()` to sort the dataframe regarding the metric.

# %%


# %% [markdown]
# ## Hyperparameters tuning 
# 
# Fine-tune the Top-k models (3 < k < 5) using a ` GridSearchCV`  (that is in sklearn.model_selection
# ) to find the best hyperparameters and achieve the maximum performance of each of the Top-k models, then compare them again to select the best one.

# %%


# %% [markdown]
# # Export key components
# Here is the section to **export** the important ML objects that will be use to develop an app: *Encoder, Scaler, ColumnTransformer, Model, Pipeline, etc*.

# %%



