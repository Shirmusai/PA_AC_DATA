# 1.1 - Installing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.cluster import KMeans
import numpy as np
import squarify
import matplotlib
matplotlib.use('TkAgg')

# 1.2 - First data file is selected: employees

# 1.3 - Print df information
df = pd.read_csv('input1_df.csv')
print("df info: ")
print(df.info())

# 1.4 - df information in Word

# 1.5 - Percentage of missing values
null_values = sum(df.isnull().sum())
sum_values = (df.shape[0]) * len(df.columns)
print()
print("Percentage of missing values: ")
print(null_values / sum_values * 100, end="%")
print()

# 1.6 - The percentage of missing values in our file is greater than 10%

# 1.7 - The file is not clean, we messed up some of the values that are listed in the correct format in the column

# 1.8 - Print of 5 rows: first, last, middle
print()
print("5 First rows:")
print(df.head())
print()
print("5 Last rows:")
print(df.tail())
print()
print("5 Middle rows:")
print(df[90:95])

# 1.9 - Descriptive statistics of the data in the df
print()
print(df.describe())
print()

# 1.10 -

# 1.2 - Second data file is selected: clothes

# 1.3 - Print df information
new_df = pd.read_csv('input2_df.csv')
print("df_1 info: ")
print(new_df.info())

# 1.4 - df information in Word

# 1.5 - Percentage of missing values
null_values = sum(new_df.isnull().sum())
sum_values = (new_df.shape[0]) * len(df.columns)
print()
print("Percentage of missing values: ")
print(null_values / sum_values * 100, end="%")
print()

# 1.6 - The percentage of missing values in our file is greater than 10%

# 1.11 - Full Outer Join
outer_join_df = df.merge(new_df, on='ID', how='outer')
print()
print("Amount of missing values:")
print(sum(outer_join_df.isnull().sum()))
print()

# 1.12 - Creating 2 new df
columns_one = ['ID', 'First Name', 'Gender', 'Salary', 'Bonus %', 'Team', 'Start Date', 'Last Login Time']
df1 = outer_join_df[columns_one]
df1 = df1.copy()

columns_two = ['ID', 'Brand', 'Category', 'Color', 'Price', 'Size', 'Material']
df2 = outer_join_df[columns_two]
df2 = df2.copy()

df1.to_csv("df1_after_join.csv", index=False)
df2.to_csv("df2_after_join.csv", index=False)

df1 = pd.read_csv("df1_after_join.csv")
df2 = pd.read_csv("df2_after_join.csv")

df2 = df2.drop(labels=range(1000, 1000), axis=0)
df2.to_csv("df2_after_join_and_delete.csv", index=False)


# 2.1 - Checks if there are string or boolean values in a numeric column
def char_finder(data_frame, series_name):
    cnt = 0
    print(series_name)
    for row in data_frame[series_name]:
        try:
            float(row)
        except ValueError:
            print(data_frame.loc[cnt, series_name], "-> at row:" + str(cnt))
        cnt += 1


print("char_finder df1 - Checks if there are string or boolean values in a numeric column:")
char_finder(df1, "Salary")
print()
char_finder(df1, "Bonus %")
print()
print("char_finder df2- Checks if there are string or boolean values in a numeric column:")
char_finder(df2, "Price")
print()


def char_fixer(data_frame, series_name):
    cnt = 0
    for row in data_frame[series_name]:
        try:
            float(row)
            pass
        except ValueError:
            data_frame.drop([cnt], inplace=True)
        cnt += 1
    data_frame[series_name] = data_frame[series_name].astype('float64', errors='raise')
    data_frame.reset_index(drop=True, inplace=True)


print()
print("char_fixer result of df1:")
char_fixer(df1, "Salary")
char_fixer(df1, "Bonus %")
print()
print(df1.dtypes)
print()
print("char_fixer result of df2:")
char_fixer(df2, "Price")
print()
print(df2.dtypes)
print()

df1.to_csv("df1_after_char_fix.csv", index=False)
df2.to_csv("df2_after_char_fix.csv", index=False)

df1 = pd.read_csv('df1_after_char_fix.csv')
df2 = pd.read_csv('df2_after_char_fix.csv')


# 2.2 - Checks if there are boolean or numeric values in a string column
def num_finder(data_frame, series_name):
    cnt = 0
    for row in data_frame[series_name]:
        try:
            int(float(row))
        except ValueError:
            if row == 'True' or row == 'False':
                print(data_frame.loc[cnt, series_name], "-> at row:" + str(cnt))
            else:
                pass
        else:
            print(data_frame.loc[cnt, series_name], "-> at row:" + str(cnt))
        cnt += 1


print("num_finder df1 - Checks if there are boolean or numeric values in a string column:")
print("Gender")
num_finder(df1, "Gender")
print()
print("Team")
num_finder(df1, "Team")
print()
print("num_finder df2- Checks if there are boolean or numeric values in a string column:")
print("Brand")
num_finder(df2, "Brand")
print()


def num_fixer(data_frame, series_name):
    cnt = 0
    for row in data_frame[series_name]:
        try:
            int(float(row))
        except ValueError:
            if row == 'True' or row == 'False':
                data_frame.drop([cnt], inplace=True)
            elif row == 'nan':
                data_frame.loc[cnt, series_name] = np.nan
            else:
                pass
        else:
            data_frame.drop([cnt], inplace=True)
        cnt += 1
    data_frame[series_name] = data_frame[series_name].astype('string', errors='raise')
    data_frame.reset_index(drop=True, inplace=True)


print()
print("num_fixer result of df1:")
num_fixer(df1, "Gender")
num_fixer(df1, "Team")
print()
print(df1.dtypes)
print()
print("num_fixer result of df2:")
num_fixer(df2, "Brand")
print()
print(df2.dtypes)
print()

df1.to_csv("df1_after_num_fix.csv", index=False)
df2.to_csv("df2_after_num_fix.csv", index=False)

df1 = pd.read_csv('df1_after_num_fix.csv')
df2 = pd.read_csv('df2_after_num_fix.csv')

# 2.3 - Replacing missing values
# df1
df1['Salary'] = pd.to_numeric(df1['Salary'], errors='coerce')
salary_mean = df1['Salary'].mean()
df1['Salary'] = df1['Salary'].fillna(salary_mean)

df1['Bonus %'] = pd.to_numeric(df1['Bonus %'], errors='coerce')
salary_mean = df1['Bonus %'].mean()
df1['Bonus %'] = df1['Bonus %'].fillna(salary_mean)

most_common_Team = df1['Team'].mode()[0]
df1['Team'] = df1['Team'].fillna(most_common_Team)

most_common_First_Name = df1['First Name'].mode()[0]
df1['First Name'] = df1['First Name'].fillna(most_common_First_Name)

most_common_Gender = df1['Gender'].mode()[0]
df1['Gender'] = df1['Gender'].fillna(most_common_Gender)

most_common_Start_Date = df1['Start Date'].mode()[0]
df1['Start Date'] = df1['Start Date'].fillna(most_common_Start_Date)

most_common_Last_Login_Time = df1['Last Login Time'].mode()[0]
df1['Last Login Time'] = df1['Last Login Time'].fillna(most_common_Last_Login_Time)

df1.to_csv("df1_change_null.csv", index=False)

# df2
df2['Price'] = pd.to_numeric(df2['Price'], errors='coerce')
Price_mean = df2['Price'].mean()
df2['Price'] = df2['Price'].fillna(Price_mean)

most_common_Brand = df2['Brand'].mode()[0]
df2['Brand'] = df2['Brand'].fillna(most_common_Brand)

most_common_Category = df2['Category'].mode()[0]
df2['Category'] = df2['Category'].fillna(most_common_Category)

most_common_Color = df2['Color'].mode()[0]
df2['Color'] = df2['Color'].fillna(most_common_Color)

most_common_Size = df2['Size'].mode()[0]
df2['Size'] = df2['Size'].fillna(most_common_Size)

most_common_Material = df2['Material'].mode()[0]
df2['Material'] = df2['Material'].fillna(most_common_Material)

df2.to_csv("df2_change_null.csv", index=False)

# 2.4 - Normalize a numeric data column
df1['Salary_norm'] = np.nan
max_Salary = df1['Salary'].max()

for key, value in df1.iterrows():
    Salary_value = value['Salary']
    Salary_norm_value = Salary_value / max_Salary
    df1.loc[df1.index[key], 'Salary_norm'] = Salary_norm_value

df1.to_csv('df1_adding_Salary_norm.csv', index=False)
df1 = pd.read_csv('df1_adding_Salary_norm.csv')


# 2.5 - Printing the duplicate lines and downloading them
def duplicate_csv_row(file_path, row_number, duplicate_times):
    rows = []
    csvfile = open(file_path, newline='')
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)
    csvfile.close()

    duplicated_row = rows[row_number - 1]
    for _ in range(duplicate_times):
        rows.insert(row_number, duplicated_row)

    csvfile = open(file_path, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerows(rows)
    csvfile.close()


df1 = pd.read_csv('df1_adding_Salary_norm.csv')
df1.to_csv('df1_creating_duplicates.csv', index=False)

file_path = "df1_creating_duplicates.csv"
row_number = 3
duplicate_times = 5

duplicate_csv_row(file_path, row_number, duplicate_times)


def find_and_print_duplicates(file_path):
    df = pd.read_csv(file_path)
    duplicate_rows = df[df.duplicated()]

    if not duplicate_rows.empty:
        print("Duplicate Rows:")
        print(duplicate_rows)
    else:
        print("No duplicate rows found.")


file_path = "df1_creating_duplicates.csv"
find_and_print_duplicates(file_path)


def drop_duplicates_csv(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.to_csv(file_path, index=False)
    return True


df1 = pd.read_csv('df1_creating_duplicates.csv')
df1.to_csv('df1_drop_duplicates.csv', index=False)

file_path2 = "df1_drop_duplicates.csv"

drop_duplicates_csv(file_path2)

# 3 - Visual presentation of the data - graphs
# 1- Bar plot
colors = plt.cm.viridis(np.linspace(0, 1, len(df1['ID'])))
plt.figure(figsize=(10, 6))
plt.bar(df1['ID'], df1['Salary'], color=colors)
plt.xlabel('ID')
plt.ylabel('Salary')
plt.title('Salaries of Employees')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2- Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df1['Salary'], df1['Bonus %'], c=df1['Bonus %'], cmap='viridis')
plt.xlabel('Salary')
plt.ylabel('Bonus %')
plt.title('Relationship between Salary and Bonus %')
plt.colorbar(label='Bonus %')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3- Pie Chart
pivot_table = df.pivot_table(index='Team', columns='Gender', aggfunc='size', fill_value=0)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightgray', 'lightpink']
for (team, data), ax in zip(pivot_table.iterrows(), axs.flatten()):
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=colors[:len(data)])
    ax.set_title(f'Team: {team}')
plt.suptitle('Distribution of Employees by Gender and Team')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 4- Area plot
gender = 'Gender'
bonus = 'Bonus %'
plt.figure(figsize=(10, 6))
plt.fill_between(df1.index, df1[gender], df1[bonus], color="skyblue", alpha=0.4)
plt.plot(df1.index, df1[gender], color="Slateblue", alpha=0.6, linewidth=2, label=gender)
plt.plot(df1.index, df1[bonus], color="Indianred", alpha=0.6, linewidth=2, label=bonus)
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Area Plot of Gender and Bonus')
plt.legend()
plt.grid(True)
plt.show()

# 5- Scatter matrix plot
variables = ['Gender', 'Salary', 'Bonus %']
pd.plotting.scatter_matrix(df1[variables], figsize=(10, 10))
plt.suptitle('Scatter Matrix Plot of Gender, Salary, and Bonus %')
plt.show()

# 6 - Box plot
x_variable = 'Brand'
y_variable = 'Price'
plt.figure(figsize=(10, 6))
sns.boxplot(x=x_variable, y=y_variable, data=df2)
plt.title('Box Plot of {} by {}'.format(y_variable, x_variable))
plt.xlabel(x_variable)
plt.ylabel(y_variable)
plt.xticks(rotation=45)
plt.show()

# 7- Line plot
x_variable = 'Color'
y_variable = 'Price'
plt.figure(figsize=(10, 6))
sns.lineplot(x=x_variable, y=y_variable, data=df2)
plt.title('Line Plot of {} by {}'.format(y_variable, x_variable))
plt.xlabel(x_variable)
plt.ylabel(y_variable)
plt.xticks(rotation=45)
plt.show()

# 8- Violin plot
variables = ['Brand', 'Category']
plt.figure(figsize=(10, 6))
sns.violinplot(x=variables[0], y=variables[1], data=df2, hue=variables[0], palette="muted", legend=False)
plt.title('Violin Plot of Selected Variables')
plt.xlabel(variables[0])
plt.ylabel(variables[1])
plt.show()

# 9- Tree map
variables = ['Category', 'Size']
data = df2[variables].value_counts()
plt.figure(figsize=(10, 6))
squarify.plot(sizes=data.values, label=data.index, alpha=0.7, color=plt.cm.Set3.colors)
plt.axis('off')
plt.title('Tree Map of Selected Variables')
plt.show()

# 10- Count plot
variables = ['Category', 'Color']
plt.figure(figsize=(10, 6))
sns.countplot(x=variables[0], hue=variables[1], data=df2, palette="Set3")
plt.title('Count Plot of Selected Variables')
plt.xlabel(variables[0])
plt.ylabel('Count')
plt.legend(title=variables[1])
plt.show()


# 4.1 - Division into clusters and presentation of the results by a graph
dataset = pd.read_csv('df1_drop_duplicates.csv')
X = dataset.iloc[:, [4, 8]] .values

wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 6), wcss)
plt.title('Elbow Method for Determining Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
y_kmeans = kmeans.fit_predict(X)
print()
print("Point by Cluster:")
print(y_kmeans)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='pink',
label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100,
c='orange', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100,
c='purple', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan',
label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100,
c='yellow', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,
1], s=200, c='black', label='Centroids')

plt.title('Clusters of students')
plt.xlabel('Bonus %')
plt.ylabel('Salary_norm')
plt.legend()
plt.show()

# 4.2 - Linear regression
x = dataset['Salary']
y = dataset['Bonus %']

a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x -
x.mean())**2))

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
ax.plot(x, y, 'o', color='tab:pink')
plt.title('Linear Fit of Salary and Bonus')
plt.xlabel('Salary')
plt.ylabel('Bonus %')
plt.show()

