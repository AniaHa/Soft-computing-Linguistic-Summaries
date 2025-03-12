import skfuzzy as fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

df = pd.read_csv(r'data_BD.csv', index_col = 0)
n_row = df.shape[0]
n_col = df.shape[1]
col_names = df.columns

'''  missing data summaries

# amount of na in each column
df.isnull().sum()
df.isnull().sum() / n_row * 100

# % of na in the whole data set
df.isnull().sum().sum() / df.size * 100

'''

# defining categorical and date variables

df["class"] = df["class"].astype("category")
df['date'] = df['date'].astype('datetime64')

''' basic statistics

stat = df.describe()
df["class"].describe()
df['class'].value_counts()

# how many different patients id's are there
df['patient_id'].unique()

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

'''

# patients with sufficiently observations
i = df['patient_id'].value_counts().index[df['patient_id'].value_counts() > 10]
df2 = df.loc[df['patient_id'].isin(list(i.values))]
df2 = df2.reset_index().drop(columns = 'index')


def trapmf_plot(x, s):
    """
    x column name (variable)
    s universum step
    y universum
    """
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


def kwantyfikator_plot(a, x, s, dfr, pid=0):
    """
    quantifier with plot
    a argument for fuzzy number
    x column name (variable)
    s universum step
    pid patient id (id = 0 when we use all patients)
    dfr data frame
    """
    if pid == 0:
        y = np.arange(x.min() - s, x.max() + 2 * s, s)

        x_l = fuzz.trapmf(y, [y.min(), y.min(), x.quantile(0.25), x.quantile(0.45)])
        x_m = fuzz.trapmf(y, [x.quantile(0.25), x.quantile(0.5), x.quantile(0.6), x.quantile(0.75)])
        x_h = fuzz.trapmf(y, [x.quantile(0.65), x.quantile(0.85), y.max(), y.max()])

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
        x_p = x.loc[dfr['patient_id'] == pid].dropna()
        if len(x_p) in (0, 1) or np.isnan(a):
            return float("nan"), float("nan"), float("nan"), x_p

        if x_p.min() == x_p.max():
            return float("nan"), float("nan"), float("nan")

        else:
            y = np.arange(x_p.min() - s, x_p.max() + 2 * s, s)

        x_l = fuzz.trapmf(y, [y.min(), y.min(), x_p.quantile(0.25), x_p.quantile(0.45)])
        x_m = fuzz.trapmf(y, [x_p.quantile(0.25), x_p.quantile(0.5), x_p.quantile(0.6), x_p.quantile(0.75)])
        x_h = fuzz.trapmf(y, [x_p.quantile(0.65), x_p.quantile(0.85), y.max(), y.max()])

        fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
        ax0.plot(y, x_l, 'b', linewidth = 1.5, label = 'Low')
        ax0.plot(y, x_m, 'g', linewidth = 1.5, label = 'Medium')
        ax0.plot(y, x_h, 'r', linewidth = 1.5, label = 'High')
        ax0.set_title('Amount in general')
        ax0.legend()

        low = fuzz.interp_membership(y, x_l, a)
        medium = fuzz.interp_membership(y, x_m, a)
        high = fuzz.interp_membership(y, x_h, a)

    return low, medium, high


def kwantyfikator(a, x, s, dfr, pid=0):
    """
    quantifier without plot
    a argument for fuzzy number
    x column name (variable)
    s universum step
    pid patient id (id = 0 when we use all patients)
    dfr data frame
    """
    if pid == 0:
        y = np.arange(x.min() - s, x.max() + 2 * s, s)

        x_l = fuzz.trapmf(y, [y.min(), y.min(), x.quantile(0.25), x.quantile(0.45)])
        x_m = fuzz.trapmf(y, [x.quantile(0.25), x.quantile(0.5), x.quantile(0.6), x.quantile(0.75)])
        x_h = fuzz.trapmf(y, [x.quantile(0.65), x.quantile(0.85), y.max(), y.max()])

        low = fuzz.interp_membership(y, x_l, a)
        medium = fuzz.interp_membership(y, x_m, a)
        high = fuzz.interp_membership(y, x_h, a)

    else:
        x_p = x.loc[dfr['patient_id'] == pid].dropna()
        if len(x_p) in (0, 1) or np.isnan(a):
            return float("nan"), float("nan"), float("nan"), x_p

        if x_p.min() == x_p.max():
            return float("nan"), float("nan"), float("nan")

        else:
            y = np.arange(x_p.min() - s, x_p.max() + 2 * s, s)

        x_l = fuzz.trapmf(y, [y.min(), y.min(), x_p.quantile(0.25), x_p.quantile(0.45)])
        x_m = fuzz.trapmf(y, [x_p.quantile(0.25), x_p.quantile(0.5), x_p.quantile(0.6), x_p.quantile(0.75)])
        x_h = fuzz.trapmf(y, [x_p.quantile(0.65), x_p.quantile(0.85), y.max(), y.max()])

        low = fuzz.interp_membership(y, x_l, a)
        medium = fuzz.interp_membership(y, x_m, a)
        high = fuzz.interp_membership(y, x_h, a)

    return low, medium, high


# new_df - data frame with degrees of fulfillment
ids = df2['patient_id'].unique()
new_col_names = col_names[5:25]

new_df = pd.DataFrame(index = list(range(954)))
new_df['PatientId'] = df2['patient_id']
for j in range(20):
    new_df['LS_low_' + new_col_names[j]] = 0.0
    new_df['LS_medium_' + new_col_names[j]] = 0.0
    new_df['LS_high_' + new_col_names[j]] = 0.0
for j in range(20):
    for i in range(954):
        k = df2['patient_id'][i]
        res = kwantyfikator(df2[new_col_names[j]][i], df2[new_col_names[j]], 0.005, df2, k)
        new_df['LS_low_' + new_col_names[j]][i] = res[0]
        new_df['LS_medium_' + new_col_names[j]][i] = res[1]
        new_df['LS_high_' + new_col_names[j]][i] = res[2]

new_col_names_np = new_df.columns[1::]  # colnames without patient id
for j in new_col_names_np:
    new_df[j] = new_df[j].apply(lambda v: 0 if v < 1e-5 else v)

# defining fuzzy number for HAMD and YMS

# HAMD
df2.hamd.max()
y = np.arange(0 - 0.005, 52 + 2 * 0.005, 0.005)
x_l = fuzz.trapmf(y, [6, 8, 12, 13])
x_m = fuzz.trapmf(y, [12, 13, 17, 18])
x_h = fuzz.trapmf(y, [17, 18, 29, 30])
x_vh = fuzz.trapmf(y, [29, 30, y.max(), y.max()])

# plot for HAMD
fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
ax0.plot(y, x_l, 'b', linewidth = 1.5, label = 'Low')
ax0.plot(y, x_m, 'g', linewidth = 1.5, label = 'Medium')
ax0.plot(y, x_h, 'r', linewidth = 1.5, label = 'High')
ax0.plot(y, x_vh, 'k', linewidth = 1.5, label = 'Very high')
ax0.set_title('Depression level')
ax0.legend()

new_df['LS_low_hamd'] = 0.0
new_df['LS_medium_hamd'] = 0.0
new_df['LS_high_hamd'] = 0.0
new_df['LS_very high_hamd'] = 0.0

# qualifier for hamd
for i in range(954):
    a = df2.loc[i, 'hamd']
    new_df['LS_low_hamd'][i] = fuzz.interp_membership(y, x_l, a)
    new_df['LS_medium_hamd'][i] = fuzz.interp_membership(y, x_m, a)
    new_df['LS_high_hamd'][i] = fuzz.interp_membership(y, x_h, a)
    new_df['LS_very high_hamd'][i] = fuzz.interp_membership(y, x_vh, a)

# YMS
df2.yms.max()
y2 = np.arange(0 - 0.005, 35 + 2 * 0.005, 0.005)
z_l = fuzz.trapmf(y2, [4, 6, 13, 14])
z_m = fuzz.trapmf(y2, [13, 14, 25, 26])
z_h = fuzz.trapmf(y2, [25, 26, y2.max(), y2.max()])

# plot for YMS
fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
ax0.plot(y2, z_l, 'b', linewidth = 1.5, label = 'Hypomania')
ax0.plot(y2, z_m, 'g', linewidth = 1.5, label = 'Mania')
ax0.plot(y2, z_h, 'r', linewidth = 1.5, label = 'Heavy mania')
ax0.set_title('Mania level')
ax0.legend()

# qualifier for yms
new_df['LS_low_yms'] = 0.0
new_df['LS_medium_yms'] = 0.0
new_df['LS_high_yms'] = 0.0

for i in range(954):
    a = df2.loc[i, 'yms']
    new_df['LS_low_yms'][i] = fuzz.interp_membership(y2, z_l, a)
    new_df['LS_medium_yms'][i] = fuzz.interp_membership(y2, z_m, a)
    new_df['LS_high_yms'][i] = fuzz.interp_membership(y2, z_h, a)

# combining new variables into new data frame
new_df2 = pd.concat([df2, new_df], axis = 1)

# fuzzy variable for patient number
x = new_df.LS_high_context_prc_nottakencalls
il_y = np.arange(0 - 0.001, 1 + 2 * 0.001, 0.001)
il_almost_no = fuzz.trapmf(il_y, [0 - 0.001, 0 - 0.001, 0.2, 0.35])
il_some = fuzz.trapmf(il_y, [0.25, 0.4, 0.5, 0.6])
il_many = fuzz.trapmf(il_y, [0.5, 0.6, 0.8, 0.9])
il_almost_all = fuzz.trapmf(il_y, [0.85, 0.95, il_y.max(), il_y.max()])

fig, ax0 = plt.subplots(nrows = 1, figsize = (8, 9))
ax0.plot(il_y, il_almost_no, 'b', linewidth = 1.5, label = 'Almost no')
ax0.plot(il_y, il_some, 'g', linewidth = 1.5, label = 'Some')
ax0.plot(il_y, il_many, 'r', linewidth = 1.5, label = 'Many')
ax0.plot(il_y, il_almost_all, 'y', linewidth = 1.5, label = 'Almost all')
ax0.set_title('Amount of patients')
ax0.legend()


def degree_of_truth(x):
    """
    :param x: column or column subset
    :return: degrees of truth for one quantifier
    """
    a = np.nanmean(x)
    almost_no = fuzz.interp_membership(il_y, il_almost_no, a)
    some = fuzz.interp_membership(il_y, il_some, a)
    many = fuzz.interp_membership(il_y, il_many, a)
    almost_all = fuzz.interp_membership(il_y, il_almost_all, a)

    return almost_no, some, many, almost_all


def degree_of_support(x):
    """
    :param x: column or column subset
    :return: degree of support for one quantifier
    """
    number = 0
    for i in x.index:
        if x[i] > 0:
            number = number + 1
    result = number / (~np.isnan(x)).sum()
    return result


# LS 1 function
def generate_short_linguistic_summaries(dfr, column, state='none'):
    """
    generates a data frame with short linguistic summaries
    :param dfr: data frame of interest
    :param column: variable (column)
    :param state: one of: 'depression', 'mania', 'euthymia', 'mixed' or none
    :return: data frame with the following columns:
        patient_degree: fuzzy number describing amount of patient
        state: if not 'none', one of the chosen
        level:  fuzzy variable describing the column (variable)
        variable: column
        degree_of_truth: degree of truth for the defined patient amount and variable level
        degree_of_support: degree of support for the variable level

    """
    if state == 'none':
        columname = '_'.join(column.name.split('_')[2:])
        level = column.name.split('_')[1]
        df_s = pd.DataFrame()
        df_s['patient_degree'] = ['almost no', 'some', 'most', 'almost all']
        df_s['level'] = np.repeat(level, 4)
        df_s['variable'] = np.repeat(columname, 4)
        df_s['degree_of_truth'] = degree_of_truth(column)
        df_s['degree_of_support'] = np.repeat(degree_of_support(column), 4)
    else:
        columname = '_'.join(column.name.split('_')[2:])
        level = column.name.split('_')[1]
        ind = np.where(dfr['class'] == state)
        df_s = pd.DataFrame()
        df_s['patient_degree'] = ['almost no', 'some', 'most', 'almost all']
        df_s['state'] = np.repeat(state, 4)
        df_s['level'] = np.repeat(level, 4)
        df_s['variable'] = np.repeat(columname, 4)
        df_s['degree_of_truth'] = degree_of_truth(column.iloc[ind])
        df_s['degree_of_support'] = np.repeat(degree_of_support(column.iloc[ind]), 4)

    return df_s


def t_norm(a, b, ntype):
    """
    calculates t-norm for param a and b
    :param ntype:
        1 - minimum
        2 - product
        3 - Lukasiewicz t-norm
    """
    if ntype == 1:
        return np.minimum(a, b)
    elif ntype == 2:
        return a * b
    elif ntype == 3:
        return np.maximum(0, a + b - 1)


def degree_of_truth_ls2(x, y, ntype):
    """
    :param x: variable responding to P quantifier
    :param y: variable responding to Q quantifier
    :param ntype:
        1 - minimum
        2 - product
        3 - Lukasiewicz t-norm
    :return: degree of truth for two quantifiers
    """
    if np.sum(x) == 0:
        return 0, 0, 0, 0
    else:
        a = np.sum(t_norm(x, y, ntype)) / np.sum(x)
        almost_no = fuzz.interp_membership(il_y, il_almost_no, a)
        some = fuzz.interp_membership(il_y, il_some, a)
        many = fuzz.interp_membership(il_y, il_many, a)
        almost_all = fuzz.interp_membership(il_y, il_almost_all, a)

        return almost_no, some, many, almost_all


def degree_of_support_ls2(x, y, ntype):
    """
    :param x: variable responding to P quantifier
    :param y: variable responding to Q quantifier
    :param ntype:
        1 - minimum
        2 - product
        3 - Lukasiewicz t-norm
    :return: degree of support for two quantifiers
    """
    number = 0
    t_norma = t_norm(x, y, ntype)
    for i in x.index:
        if t_norma[i] > 0:
            number = number + 1
    return number / (~np.isnan(x)).sum()


def degree_of_focus(x):
    """
    :param x: column of interest (variable)
    :return: degree of focus
    """
    return np.sum(x) / (~np.isnan(x)).sum()


# LS 2 function
def generate_extended_linguistic_summaries(dfr, column, ntype, state):
    """
    :param dfr: data frame of interest
    :param column: variable of interest
    :param ntype: t-norm type:
                    1 - minimum
                    2 - product
                    3 - Lukasiewicz t-norm
    :param state: one of: 'depression', 'mania', 'both'
    :return: data frame with the following columns:
            patient_degree: fuzzy number describing amount of patient
            state_level: chosen state with the described by a fuzzy number
            variable: column
            level: fuzzy variable describing the column (variable)
            degree_of_truth: degree of truth for state and variable quantifier
            degree_of_support: degree of support for state and variable quantifier
            degree_of_focus: degree of truth for state quantifier
    """
    columname = '_'.join(column.name.split('_')[2:])
    level = column.name.split('_')[1]
    df_s = pd.DataFrame()
    if state == 'depression':
        depr_var = ['LS_low_hamd', 'LS_medium_hamd', 'LS_high_hamd', 'LS_very high_hamd']
        depr_var2 = [depr_var[d].split('_')[1] for d in range(4)]
        df_s['patient_degree'] = ['almost no', 'some', 'most', 'almost all'] * 4
        df_s['state_level'] = np.repeat([sub1 + ' depression' for sub1 in depr_var2], 4)
        df_s['variable'] = np.repeat(columname, 16)
        df_s['level'] = np.repeat(level, 16)
        df_s['degree_of_truth'] = np.repeat(0, 16)
        df_s['degree_of_support'] = np.repeat(0, 16)
        df_s['degree_of_focus'] = np.repeat(0, 16)
        for i in range(4):
            x = dfr[depr_var[i]]
            df_s['degree_of_truth'][4 * i:4 * i + 4] = degree_of_truth_ls2(x, column, ntype)
            df_s['degree_of_support'][4 * i:4 * i + 4] = np.repeat(degree_of_support_ls2(x, column, ntype), 4)
            df_s['degree_of_focus'][4 * i:4 * i + 4] = np.repeat(degree_of_focus(x), 4)
        return df_s
    elif state == 'mania':
        man_var = ['LS_low_yms', 'LS_medium_yms', 'LS_high_yms']
        man_var2 = [man_var[d].split('_')[1] for d in range(3)]
        df_s['patient_degree'] = ['almost no', 'some', 'most', 'almost all'] * 3
        df_s['state_level'] = np.repeat([sub1 + ' mania' for sub1 in man_var2], 4)
        df_s['variable'] = np.repeat(columname, 12)
        df_s['level'] = np.repeat(level, 12)
        df_s['degree_of_truth'] = np.repeat(0, 12)
        df_s['degree_of_support'] = np.repeat(0, 12)
        df_s['degree_of_focus'] = np.repeat(0, 12)
        for i in range(3):
            x = dfr[man_var[i]]
            df_s['degree_of_truth'][4 * i:4 * i + 4] = degree_of_truth_ls2(x, column, ntype)
            df_s['degree_of_support'][4 * i:4 * i + 4] = np.repeat(degree_of_support_ls2(x, column, ntype), 4)
            df_s['degree_of_focus'][4 * i:4 * i + 4] = np.repeat(degree_of_focus(x), 4)
        return df_s

    else:
        depr_var = ['LS_low_hamd', 'LS_medium_hamd', 'LS_high_hamd', 'LS_very high_hamd']
        depr_var2 = [depr_var[d].split('_')[1] for d in range(4)]
        man_var = ['LS_low_yms', 'LS_medium_yms', 'LS_high_yms']
        man_var2 = [man_var[d].split('_')[1] for d in range(3)]
        df_s['patient_degree'] = ['almost no', 'some', 'most', 'almost all'] * 12
        df_s['state_level'] = np.repeat(
            [sub1 + ' depression and ' + sub2 + ' mania' for sub1, sub2 in product(depr_var2, man_var2)], 4)
        df_s['variable'] = np.repeat(columname, 48)
        df_s['level'] = np.repeat(level, 48)
        df_s['degree_of_truth'] = np.repeat(0, 48)
        df_s['degree_of_support'] = np.repeat(0, 48)
        df_s['degree_of_focus'] = np.repeat(0, 48)
        for i in range(12):
            x1 = dfr[man_var[i % 3]]
            x2 = dfr[depr_var[i % 4]]
            df_s['state_level'][4 * i:4 * i + 4] = np.repeat(
                [depr_var2[i % 4] + ' depression and ' + man_var2[i % 3] + ' mania'], 4)
            x = t_norm(x1, x2, ntype)
            df_s['degree_of_truth'][4 * i:4 * i + 4] = degree_of_truth_ls2(x, column, ntype)
            df_s['degree_of_support'][4 * i:4 * i + 4] = np.repeat(degree_of_support_ls2(x, column, ntype), 4)
            df_s['degree_of_focus'][4 * i:4 * i + 4] = np.repeat(degree_of_focus(x), 4)
        return df_s


# generating all possible sentences

# LS 1
# with state defined
col_number_ls1 = generate_short_linguistic_summaries(new_df2, new_df2.LS_low_context_no_day_incoming_calls,
                                                     'depression')

result_ls1_state = pd.DataFrame(columns = col_number_ls1.columns)

for j in ['depression', 'mania', 'euthymia', 'mixed']:
    for i in range(28, new_df2.shape[1]):
        variable = new_df2.columns[i]
        result_ls1_state = result_ls1_state.append(generate_short_linguistic_summaries(new_df2, new_df2[variable], j))

result_ls1_state = result_ls1_state.reset_index(drop = True)

n = result_ls1_state.shape[0]
ls1_state_sentences = pd.DataFrame(columns = ['Sentence', 'Degree of Truth', 'Degree of Support'],
                                   index = range(n))

for i in range(n):
    sentence = ('For ' + result_ls1_state['patient_degree'][i] + ' records in ' + result_ls1_state['state'][i] + ', ' +
                result_ls1_state['variable'][i] + ' is ' + result_ls1_state['level'][i])
    ls1_state_sentences['Sentence'][i] = sentence
    ls1_state_sentences['Degree of Truth'][i] = result_ls1_state['degree_of_truth'][i]
    ls1_state_sentences['Degree of Support'][i] = result_ls1_state['degree_of_support'][i]

# sentences with degree of truth > 0
ls1_state_sentences_final = ls1_state_sentences.iloc[np.where(ls1_state_sentences['Degree of Truth'] > 0)]
# ls1_state_sentences_final.to_csv('ls1_state_sentences.csv', index = False, sep = ';')

# with state no defined

col_number_ls1_no_state = generate_short_linguistic_summaries(new_df2, new_df2.LS_low_context_no_day_incoming_calls)
result_ls1_no_state = pd.DataFrame(columns = col_number_ls1_no_state.columns)
for i in range(28, new_df2.shape[1]):
    variable = new_df2.columns[i]
    result_ls1_no_state = result_ls1_no_state.append(generate_short_linguistic_summaries(new_df2, new_df2[variable]))
result_ls1_no_state = result_ls1_no_state.reset_index(drop = True)

n = result_ls1_no_state.shape[0]
ls1_no_state_sentences = pd.DataFrame(columns = ['Sentence', 'Degree of Truth', 'Degree of Support'],
                                      index = range(n))
for i in range(n):
    sentence = ('For ' + result_ls1_no_state['patient_degree'][i] + ' records ' +
                result_ls1_no_state['variable'][i] + ' is ' + result_ls1_no_state['level'][i])
    ls1_no_state_sentences['Sentence'][i] = sentence
    ls1_no_state_sentences['Degree of Truth'][i] = result_ls1_no_state['degree_of_truth'][i]
    ls1_no_state_sentences['Degree of Support'][i] = result_ls1_no_state['degree_of_support'][i]

# sentences with degree of truth > 0
ls1_no_state_sentences_final = ls1_no_state_sentences.iloc[np.where(ls1_no_state_sentences['Degree of Truth'] > 0)]
# ls1_no_state_sentences_final.to_csv('ls1_no_state_sentences.csv', index = False)

# LS 2

col_number_ls2 = generate_extended_linguistic_summaries(new_df2, new_df2.LS_low_context_no_day_incoming_calls, 2,
                                                        'mania')
# minimum norm
result_ls2_1norm = pd.DataFrame(columns = col_number_ls2.columns)
for j in ['depression', 'mania', 'both']:
    for i in range(28, new_df2.shape[1] - 7):
        variable = new_df2.columns[i]
        result_ls2_1norm = result_ls2_1norm.append(generate_extended_linguistic_summaries(new_df2, new_df2[variable],
                                                                                          1, j))

result_ls2_1norm = result_ls2_1norm.reset_index(drop = True)

n = result_ls2_1norm.shape[0]
ls2_sentences_1norm = pd.DataFrame(columns = ['Sentence', 'Degree of Truth', 'Degree of Support', 'Degree of Focus'],
                                   index = range(n))

for i in range(n):
    sentence = ('For ' + result_ls2_1norm['patient_degree'][i] + ' records in ' + result_ls2_1norm['state_level'][i]
                + ', ' + result_ls2_1norm['variable'][i] + ' is ' + result_ls2_1norm['level'][i])
    ls2_sentences_1norm['Sentence'][i] = sentence
    ls2_sentences_1norm['Degree of Truth'][i] = result_ls2_1norm['degree_of_truth'][i]
    ls2_sentences_1norm['Degree of Support'][i] = result_ls2_1norm['degree_of_support'][i]
    ls2_sentences_1norm['Degree of Focus'][i] = result_ls2_1norm['degree_of_focus'][i]

ls2_sentences_1norm_final = ls2_sentences_1norm.iloc[np.where(ls2_sentences_1norm['Degree of Truth'] > 0)]
# ls2_sentences_1norm_final.to_csv('ls2_sentences_1norm.csv', index = False, sep=';')

# product norm
result_ls2_2norm = pd.DataFrame(columns = col_number_ls2.columns)
for j in ['depression', 'mania', 'both']:
    for i in range(28, new_df2.shape[1] - 7):
        variable = new_df2.columns[i]
        result_ls2_2norm = result_ls2_2norm.append(generate_extended_linguistic_summaries(new_df2, new_df2[variable],
                                                                                          2, j))

result_ls2_2norm = result_ls2_2norm.reset_index(drop = True)

n = result_ls2_2norm.shape[0]
ls2_sentences_2norm = pd.DataFrame(columns = ['Sentence', 'Degree of Truth', 'Degree of Support', 'Degree of Focus'],
                                   index = range(n))

for i in range(n):
    sentence = ('For ' + result_ls2_2norm['patient_degree'][i] + ' records in ' + result_ls2_2norm['state_level'][i]
                + ', ' + result_ls2_2norm['variable'][i] + ' is ' + result_ls2_2norm['level'][i])
    ls2_sentences_2norm['Sentence'][i] = sentence
    ls2_sentences_2norm['Degree of Truth'][i] = result_ls2_2norm['degree_of_truth'][i]
    ls2_sentences_2norm['Degree of Support'][i] = result_ls2_2norm['degree_of_support'][i]
    ls2_sentences_2norm['Degree of Focus'][i] = result_ls2_2norm['degree_of_focus'][i]

ls2_sentences_2norm_final = ls2_sentences_2norm.iloc[np.where(ls2_sentences_2norm['Degree of Truth'] > 0)]
# ls2_sentences_2norm_final.to_csv('ls2_sentences_2norm.csv', index = False, sep=';')


# Lukasiewicz norm
result_ls2_3norm = pd.DataFrame(columns = col_number_ls2.columns)
for j in ['depression', 'mania', 'both']:
    for i in range(28, new_df2.shape[1] - 7):
        variable = new_df2.columns[i]
        result_ls2_3norm = result_ls2_3norm.append(generate_extended_linguistic_summaries(new_df2, new_df2[variable],
                                                                                          2, j))

result_ls2_3norm = result_ls2_3norm.reset_index(drop = True)

n = result_ls2_3norm.shape[0]
ls2_sentences_3norm = pd.DataFrame(columns = ['Sentence', 'Degree of Truth', 'Degree of Support', 'Degree of Focus'],
                                   index = range(n))

for i in range(n):
    sentence = ('For ' + result_ls2_3norm['patient_degree'][i] + ' records in ' + result_ls2_3norm['state_level'][i]
                + ', ' + result_ls2_3norm['variable'][i] + ' is ' + result_ls2_3norm['level'][i])
    ls2_sentences_3norm['Sentence'][i] = sentence
    ls2_sentences_3norm['Degree of Truth'][i] = result_ls2_3norm['degree_of_truth'][i]
    ls2_sentences_3norm['Degree of Support'][i] = result_ls2_3norm['degree_of_support'][i]
    ls2_sentences_3norm['Degree of Focus'][i] = result_ls2_3norm['degree_of_focus'][i]

ls2_sentences_3norm_new = ls2_sentences_3norm.iloc[np.where(ls2_sentences_3norm['Degree of Truth'] > 0)]
ls2_sentences_3norm_final = ls2_sentences_3norm_new.iloc[np.where(ls2_sentences_3norm_new['Degree of Support'] > 0)]
# ls2_sentences_3norm_final.to_csv('ls2_sentences_3norm.csv', index = False, sep=';')

# creating one data frame with all sentences
ls1_no_state_sentences['Degree of Focus'] = np.repeat(0, 268)
ls1_state_sentences['Degree of Focus'] = np.repeat(0, 1072)

all_sentences_ls1 = pd.concat([ls1_state_sentences, ls1_no_state_sentences], axis = 0)

all_sentences_ls1_new = all_sentences_ls1.iloc[np.where((all_sentences_ls1['Degree of Truth'] > 0.45) &
                                                        (all_sentences_ls1['Degree of Support'] > 0.35) &
                                                        (all_sentences_ls1['Sentence'].str.find(' some ') == -1))]
all_sentences_ls1_new = all_sentences_ls1_new.drop_duplicates()

all_sentences_ls2_1 = ls2_sentences_1norm.iloc[np.where((ls2_sentences_1norm['Degree of Truth'] != 0) &
                                                      (ls2_sentences_1norm['Degree of Support'] != 0) &
                                                      (ls2_sentences_1norm['Sentence'].str.find(' some ') == -1) &
                                                      (ls2_sentences_1norm['Sentence'].str.find(' almost no ') == -1))]
all_sentences_ls2_1 = all_sentences_ls2_1.drop_duplicates()
all_sentences_ls2_2 = ls2_sentences_1norm.iloc[np.where((ls2_sentences_1norm['Degree of Truth'] > 0.4) &
                                                      (ls2_sentences_1norm['Degree of Support'] > 0.054) &
                                                      (ls2_sentences_1norm['Sentence'].str.find(' some ') == -1) &
                                                      (ls2_sentences_1norm['Sentence'].str.find(' almost no ') != -1))]
all_sentences_ls2_2 = all_sentences_ls2_2.drop_duplicates()

# deleting sentences with no reasonable interpretation

all_sentences_ls2_2 = all_sentences_ls2_2.drop([512, 496], axis = 0)
all_sentences_ls1_new = all_sentences_ls1_new.drop([526, 1046], axis = 0)

all_sentences_final = pd.concat([all_sentences_ls1_new, all_sentences_ls2_1, all_sentences_ls2_2], axis = 0)
# all_sentences_final.to_csv('all_sentences_final.csv', index = False, sep=';')