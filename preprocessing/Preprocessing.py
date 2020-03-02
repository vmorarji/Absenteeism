import pandas as pd



raw_csv_data = pd.read_csv('Absenteeism-data.csv')
df = raw_csv_data.copy()
df.drop(['ID'],inplace=True, axis=1)


## Get dummies for the reason for absence column
reason_columns = pd.get_dummies(df['Reason for Absence'],drop_first=True)
df.drop(['Reason for Absence'], axis=1, inplace=True)



## Split the reason for absence into four basic categories. 
## type_1 contains various disease related reasons for absence
## type_2 contains all reasons related to pregnancy
## type_3 contains reasons related to poisoning
## type_4 relates to 'light' reasons for absence such as doctors appointments

reason_type_1 = reason_columns.loc[:,0:14].max(axis=1)
reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:,22:].max(axis=1)


df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3,reason_type_4], axis=1)


column_names=  ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']


df.columns = column_names


## reorder the df
df = df[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']]


df_reason_mod = df.copy()


## change the data column from a string to a datetime
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'],format="%d/%m/%Y")



## extract the month from the date and create a new column called 'Month Value'
list_months = []

for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)
df_reason_mod['Month Value'] = list_months


## extract the day of the week from the date and create a new column called 'Day of the Week'
def date_to_weekday(date_value):
    return date_value.weekday()

df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)
df_reason_mod.drop(['Date'],inplace=True,axis=1)


df_reason_mod = df_reason_mod[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
       'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
       'Pets', 'Absenteeism Time in Hours']]

df_reason_date_mod = df_reason_mod.copy()


## reason 1 is upto highschool
## reasons 2,3,4 are graduate, postgraduate and masters or a docorate respectively
## so in a binary classification value 0 will be highschool and value 1 will be further education
df_reason_date_mod['Education'].value_counts()


df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})



df_preprocessed = df_reason_date_mod.copy()


df_preprocessed.to_csv('df_preprocessed.csv', index=False)
