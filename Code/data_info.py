import skfuzzy as fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'data_BD.csv', index_col = 0)
print(df)
n_row = df.shape[0]
n_col = df.shape[1]
col_names = df.columns

# amount of na in each column
df.isnull().sum()
df.isnull().sum() / n_row * 100

# % of na in the whole data set
df.isnull().sum().sum() / df.size * 100

# defining categorical and date variables
df["class"] = df["class"].astype("category")
df['date'] = df['date'].astype('datetime64')

# basic statistics
stat = df.describe()
df["class"].describe()
df['class'].value_counts()

# how many different patients id's are there
df['patient_id'].nunique()
# how many of each
df['patient_id'].value_counts()

# start and end date
df['date'].min()
df['date'].max()
# 438 - days data
df['date'].max() - df['date'].min()

# observations for each day
df.groupby(df['date']).count()['patient_id'].plot()

df.groupby(df['date']).mean()

# dividing by class
euthymia = df[df['class'] == 'euthymia']
depression = df[df['class'] == 'depression']
mania = df[df['class'] == 'mania']
mixed = df[df['class'] == 'mixed']

# boxplots
# sns.boxplot(x = df['class'], y = df['context_avg_day_duration_incoming'])
# sns.boxplot(x = df['class'], y = df['context_no_day_incoming_calls'])

# od Antka

if __name__ == '__main__':
    dane = df
    print(dane.info())
    sns.boxplot(y = 'context_avg_day_duration_incoming', x = 'patient_id', data = dane)
    plt.show()
    sns.boxplot(y = 'yms', x = 'patient_id', data = dane)
    plt.show()
    sns.boxplot(y = 'hamd', x = 'patient_id', data = dane)
    plt.show()

# patients with less or equal to 3 observations
i = df['patient_id'].value_counts().index[df['patient_id'].value_counts() > 4]
df2 = df.loc[df['patient_id'].isin(list(i.values))]
df2 = df2.reset_index().drop(columns = 'index')
df2.index = df2.index + 1
# trapmf, interp_membership (w jakim stopniu dany punkt przynalezy do rozmytego)

df2['yms'].describe()
x = np.arange(df2.yms.min(), df2.yms.max(), 0.01)

yms_l = fuzz.trapmf(x, [df2.yms.min(), df2.yms.min(), df2.yms.mean(), df2.yms.mean()])
yms_m = fuzz.trapmf(x, [df2.yms.quantile(0.5), df2.yms.quantile(0.5), df2.yms.quantile(0.75), df2.yms.quantile(0.75)])
yms_h = fuzz.trapmf(x, [df2.yms.quantile(0.75), df2.yms.quantile(0.75), df2.yms.max(), df2.yms.max()])


# fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
# ax0.plot(x, yms_l, 'b', linewidth = 1.5, label = 'Low')
# ax0.plot(x, yms_m, 'g', linewidth = 1.5, label = 'Medium')
# ax0.plot(x, yms_h, 'r', linewidth = 1.5, label = 'High')
# ax0.set_title('Height')
# ax0.legend()


def trapmf_plot(x, s):
    y = np.arange(x.min(), x.max(), s)

    x_l = fuzz.trapmf(y, [x.min(), x.min(), x.quantile(0.25), x.quantile(0.45)])
    x_m = fuzz.trapmf(y, [x.quantile(0.25), x.quantile(0.5), x.quantile(0.6), x.quantile(0.75)])
    x_h = fuzz.trapmf(y, [x.quantile(0.65), x.quantile(0.85), x.max(), x.max()])

    fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
    ax0.plot(y, x_l, 'b', linewidth = 1.5, label = 'Low')
    ax0.plot(y, x_m, 'g', linewidth = 1.5, label = 'Medium')
    ax0.plot(y, x_h, 'r', linewidth = 1.5, label = 'High')
    ax0.set_title('Height')
    ax0.legend()


# trapmf_plot(df2.context_avg_day_duration_incoming, 1)


def kwantyfikator_plot(a, x, s, id=0):
    if id == 0:
        y = np.arange(x.min(), x.max(), s)

        x_l = fuzz.trapmf(y, [0, x.min(), x.quantile(0.25), x.quantile(0.45)])
        x_m = fuzz.trapmf(y, [x.quantile(0.25), x.quantile(0.5), x.quantile(0.6), x.quantile(0.75)])
        x_h = fuzz.trapmf(y, [x.quantile(0.65), x.quantile(0.85), x.max(), x.max()])

        fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
        ax0.plot(y, x_l, 'b', linewidth = 1.5, label = 'Low')
        ax0.plot(y, x_m, 'g', linewidth = 1.5, label = 'Medium')
        ax0.plot(y, x_h, 'r', linewidth = 1.5, label = 'High')
        ax0.set_title('Amount in general')
        ax0.legend()

        low = fuzz.interp_membership(y, x_l, a)
        medium = fuzz.interp_membership(y, x_m, a)
        high = fuzz.interp_membership(y, x_h, a)

    else:
        x_p = x.loc[df['patient_id'] == id]
        y = np.arange(x_p.min(), x_p.max(), s)

        x_l = fuzz.trapmf(y, [0, x_p.min(), x_p.quantile(0.25), x_p.quantile(0.45)])
        x_m = fuzz.trapmf(y, [x_p.quantile(0.25), x_p.quantile(0.5), x_p.quantile(0.6), x_p.quantile(0.75)])
        x_h = fuzz.trapmf(y, [x_p.quantile(0.65), x_p.quantile(0.85), x_p.max(), x_p.max()])

        fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
        ax0.plot(y, x_l, 'b', linewidth = 1.5, label = 'Low')
        ax0.plot(y, x_m, 'g', linewidth = 1.5, label = 'Medium')
        ax0.plot(y, x_h, 'r', linewidth = 1.5, label = 'High')
        ax0.set_title('Amount for individual')
        ax0.legend()

        low = fuzz.interp_membership(y, x_l, a)
        medium = fuzz.interp_membership(y, x_m, a)
        high = fuzz.interp_membership(y, x_h, a)

    return low, medium, high


def kwantyfikator(a, x, s, id=0):
    if id == 0:
        y = np.arange(x.min(), x.max(), s)

        x_l = fuzz.trapmf(y, [x.min(), x.min(), x.quantile(0.25), x.quantile(0.45)])
        x_m = fuzz.trapmf(y, [x.quantile(0.25), x.quantile(0.5), x.quantile(0.6), x.quantile(0.75)])
        x_h = fuzz.trapmf(y, [x.quantile(0.65), x.quantile(0.85), x.max(), x.max()])

        low = fuzz.interp_membership(y, x_l, a)
        medium = fuzz.interp_membership(y, x_m, a)
        high = fuzz.interp_membership(y, x_h, a)

    else:
        x_p = x.loc[df2['patient_id'] == id].dropna()
        if len(x_p) == 0:
            raise Warning("x_p is empty!!! :((")
        y = np.arange(x_p.min(), x_p.max(), s)

        x_l = fuzz.trapmf(y, [0, x_p.min(), x_p.quantile(0.25), x_p.quantile(0.45)])
        x_m = fuzz.trapmf(y, [x_p.quantile(0.25), x_p.quantile(0.5), x_p.quantile(0.6), x_p.quantile(0.75)])
        x_h = fuzz.trapmf(y, [x_p.quantile(0.65), x_p.quantile(0.85), x_p.max(), x_p.max()])

        low = fuzz.interp_membership(y, x_l, a)
        medium = fuzz.interp_membership(y, x_m, a)
        high = fuzz.interp_membership(y, x_h, a)

    return low, medium, high, x_p


# kwantyfikator(4, df2.context_no_day_outgoing_calls, 1, id = 9829)
# kwantyfikator_plot(4, df2.context_no_day_outgoing_calls, 1, id = 9829)
# kwantyfikator_plot(4, df2.context_no_day_outgoing_calls, 1)

# kwantyfikator_plot(50, df2.context_avg_len_sms, 0.25, id = 9829)
# kwantyfikator_plot(50, df2.context_avg_len_sms, 1)

# zad 3
# normalization, standardization
ids = df2['patient_id'].unique()
df2 = df2.reset_index().drop(columns = 'index')
df2.index = df2.index + 1
new_col_names = col_names[5:25]
new_df = pd.DataFrame(index = list(range(1, 955)))
new_df['PatientId'] = df2['patient_id']
for j in range(20):
    new_df['LS_low_' + new_col_names[j]] = 0.0
    new_df['LS_medium_' + new_col_names[j]] = 0.0
    new_df['LS_high_' + new_col_names[j]] = 0.0
for j in range(20):
    for i in range(1, 955):
        k = df2['patient_id'][i]
        res = kwantyfikator(df2[new_col_names[j]][i], df2[new_col_names[j]], 0.0025, k)
        new_df['LS_low_' + new_col_names[j]][i] = res[0]
        new_df['LS_medium_' + new_col_names[j]][i] = res[1]
        new_df['LS_high_' + new_col_names[j]][i] = res[2]

new_df2 = pd.concat([df2, new_df], axis = 1)

y = np.arange(0, 52, 0.25)
x = df2.hamd
x_l = fuzz.trapmf(y, [6, 8, 12, 13])
x_m = fuzz.trapmf(y, [12, 13, 17, 18])
x_h = fuzz.trapmf(y, [17, 18, 29, 30])
x_vh = fuzz.trapmf(y, [29, 30, 52, 52])

fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
ax0.plot(y, x_l, 'b', linewidth = 1.5, label = 'Lagodna')
ax0.plot(y, x_m, 'g', linewidth = 1.5, label = 'Umiarkowana')
ax0.plot(y, x_h, 'r', linewidth = 1.5, label = 'Ciezka')
ax0.plot(y, x_vh, 'k', linewidth = 1.5, label = 'Bardzo ciezka')
ax0.set_title('Poziom depresji')
ax0.legend()

y = np.arange(0, 35, 0.25)
z = df2.yms
z_l = fuzz.trapmf(y, [4, 6, 13, 14])
z_m = fuzz.trapmf(y, [13, 14, 25, 26])
z_h = fuzz.trapmf(y, [25, 26, z.max(), z.max()])

fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
ax0.plot(y, z_l, 'b', linewidth = 1.5, label = 'Hipomania')
ax0.plot(y, z_m, 'g', linewidth = 1.5, label = 'Mania')
ax0.plot(y, z_h, 'r', linewidth = 1.5, label = 'Ciezka mania')
ax0.set_title('Poziom manii')
ax0.legend()
