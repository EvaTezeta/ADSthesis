import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu
from tabulate import tabulate

# Load dataset
data = pd.read_csv('C:/Users/j.perdeck/BC_Prediction/Eva/3_PredictedData_Students.csv')

# Remove unused variables from dataset
df_pre = data.drop(['Probability','Prediction'], axis=1)

# Drop columns that contain "_measured" in their name
columns_to_drop = [col for col in df_pre.columns if '_measured' in col]
df = df_pre.drop(columns=columns_to_drop)

# Drop rows with missing data
df.dropna(inplace=True)

# Separate the independent variables and the outcome
X = df.drop(columns=['outcome'])
y = df['outcome']

# Check for normality of each continuous variable
normality = {}
for column in X.columns:
	stat, p_val = shapiro(X[column])
	normality[column] = p_val

# Split data into groups based on the outcome variable
group1 = X[y == 0]  # Group with outcome 0 (negative)
group2 = X[y == 1]  # Group with outcome 1 (positive)

# Perform Mann-Whitney U test for each independent variable
p_values = {}
for column in X.columns:
	stat, p_val = mannwhitneyu(group1[column], group2[column])
	p_values[column] = p_val

# Create a table of all variables along with their p-values
table_data = {'Variable': X.columns, 'P-Value': [p_values[var] for var in X.columns]}
table_df = pd.DataFrame(table_data)

# Print the table
print(tabulate(table_df, headers='keys', tablefmt='grid'))

# Construct Pearson correlation matrix for all features
correlation_matrix = X.corr()

# Visualize correlation matrix using heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Matrix for All Features")
plt.savefig("plot_pearson_matrix.png")
plt.show()
