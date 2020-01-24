"""Displays ABS data by matching electorate count based on data denial representation of that electorate

This product (abs.py) incorporates data that is:
    © Commonwealth of Australia (Australian Electoral Commission) 2020
    © Commonwealth of Australia (Australian Bureau of Statistics) 2020

"""

import pandas as pd
import numpy as np
import math as math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

__author__ = "Andrew Arch"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Andrew Arch"
__email__ = "andy.arch11@gmail.com"
__status__ = "Production"

def climate_label (row):
    """Create a user friendly name for the Climate Change Support value
    
    Args:
        row (tuple): row to determine the friendly value for

    Returns:
        str: User Friendly name for value
    """

    if row['Climate Change Support'] == 0:
        return 'Active Climate Denier'
    elif row['Climate Change Support'] == 1:
        return 'Climate Denier'
    elif row['Climate Change Support'] == 2:
        return 'Fence Sitter'
    elif row['Climate Change Support'] == 3:
        return 'Accepts the Science'

#determine what to show
show_total = False
show_all_deniers = False
show_types_of_deniers = True
show_accepts_the_science = True
data_category = 0 #which data set use - 0 to 8 ['resident', 'age', 'family', 'income', 'tenure', 'cultural diversity', 'employment', 'occupation', 'education']

#bin boundaries per dataset
#start, stop, steps or increment values, format type
resident_B_bin = [100000.0, 225000.0, 24, 'steps', 'int']
resident_bins = [resident_B_bin]
age_B_bin = [0.06, 0.32, 0.02, 'increment', 'perc']
age_C_bin = [0.12, 0.42, 0.02, 'increment', 'perc']
age_D_bin = [0.12, 0.28, 0.02, 'increment', 'perc']
age_E_bin = [0.08, 0.26, 0.02, 'increment', 'perc']
age_F_bin = [0.02, 0.24, 0.02, 'increment', 'perc']
age_G_bin = [0.00, 0.08, 0.01, 'increment', 'perc']
age_bins = [age_B_bin, age_C_bin, age_D_bin, age_E_bin, age_F_bin, age_G_bin]
family_B_bin = [0.18, 0.66, 0.02, 'increment', 'perc']
family_C_bin = [0.20, 0.62, 0.02, 'increment', 'perc']
family_D_bin = [0.08, 0.28, 0.02, 'increment', 'perc']
family_E_bin = [0.0, 0.08, 0.01, 'increment', 'perc']
family_bins = [family_B_bin, family_C_bin, family_D_bin, family_E_bin]
income_B_bin = [800, 2500, 100, 'increment', 'dollar']
income_C_bin = [75, 650, 25, 'increment', 'dollar']
income_D_bin = [900, 3100, 100, 'increment', 'dollar']
income_bins = [income_B_bin, income_C_bin, income_D_bin]
tenure_F_bin = [0.10, 0.48, 0.02, 'increment', 'perc']
tenure_G_bin = [0.14, 0.60, 0.02, 'increment', 'perc']
tenure_H_bin = [0.12, 0.66, 0.02, 'increment', 'perc']
tenure_I_bin = [0.0, 0.04, 0.01, 'increment', 'perc']
tenure_J_bin = [0.0, 0.06, 0.01, 'increment', 'perc']
tenure_bins = [tenure_F_bin, tenure_G_bin, tenure_H_bin, tenure_I_bin, tenure_J_bin]
culture_B_bin = [0.0, 0.44, 0.01, 'increment', 'perc']
culture_C_bin = [0.02, 0.56, 0.02, 'increment', 'perc']
culture_D_bin = [0.0, 0.32, 0.01, 'increment', 'perc']
culture_E_bin = [0.0, 0.62, 0.01, 'increment', 'perc']
culture_bins = [culture_B_bin, culture_C_bin, culture_D_bin, culture_E_bin]
employment_B_bin = [0.36, 0.70, 0.02, 'increment', 'perc']
employment_C_bin = [0.06, 0.28, 0.02, 'increment', 'perc']
employment_D_bin = [0.01, 0.06, 0.01, 'increment', 'perc']
employment_E_bin = [0.08, 0.34, 0.02, 'increment', 'perc']
employment_F_bin = [0.0, 0.2, 0.02, 'increment', 'perc']
employment_bins = [employment_B_bin, employment_C_bin, employment_D_bin, employment_E_bin, employment_F_bin]
occupation_B_bin = [0.04, 0.24, 0.02, 'increment', 'perc']
occupation_C_bin = [0.08, 0.44, 0.02, 'increment', 'perc']
occupation_D_bin = [0.02, 0.22, 0.02, 'increment', 'perc']
occupation_E_bin = [0.04, 0.18, 0.02, 'increment', 'perc']
occupation_F_bin = [0.06, 0.22, 0.02, 'increment', 'perc']
occupation_G_bin = [0.02, 0.16, 0.02, 'increment', 'perc']
occupation_H_bin = [0.00, 0.18, 0.02, 'increment', 'perc']
occupation_I_bin = [0.00, 0.22, 0.02, 'increment', 'perc']
occupation_J_bin = [0.0, 0.03, 0.01, 'increment', 'perc']
occupation_K_bin = [0.0, 0.03, 0.01, 'increment', 'perc']
occupation_bins = [occupation_B_bin, occupation_C_bin, occupation_D_bin, occupation_E_bin, occupation_F_bin, occupation_G_bin, occupation_H_bin, occupation_I_bin, occupation_J_bin, occupation_K_bin]
education_B_bin = [0.3, 1.00, 0.05, 'increment', 'perc']
education_C_bin = [0.25, 0.70, 0.05, 'increment', 'perc']
education_bins = [education_B_bin, education_C_bin]

#load abs data
abs_category = ['resident', 'age', 'family', 'income', 'tenure', 'cultural diversity', 'employment', 'occupation', 'education']
abs_excel_metadata = {'Sheet': ['Table 1', 'Table 2', 'Table 3', 'Table 4', 'Table 4', 'Table 5', 'Table 6', 'Table 7', 'Table 8'],
             'Header Row': [0, 0, 0, 0, 1, 0, 0, 0, 0],
             'Columns': ['A:B', 'A:G', 'A:E', 'A:D', 'A,F:J', 'A:E', 'A:F', 'A:K', 'A:C'],
             'Number Rows': [152, 152, 152, 153, 152, 152, 152, 152, 152],
             'Bins': [resident_bins, age_bins, family_bins, income_bins, tenure_bins, culture_bins, employment_bins, occupation_bins, education_bins]}

abs_df_metadata = pd.DataFrame(data=abs_excel_metadata, index=abs_category)
abs_file = './data/commonwealth electorate data.xls'
index_column = 0
skip_rows = 5

abs_cat = abs_category[data_category]
abs_data = pd.read_excel(abs_file, sheet_name=abs_df_metadata.loc[abs_cat]['Sheet'], header=abs_df_metadata.loc[abs_cat]['Header Row'], usecols=abs_df_metadata.loc[abs_cat]['Columns'], skiprows=skip_rows, nrows=abs_df_metadata.loc[abs_cat]['Number Rows'], index_col=index_column)
abs_data.dropna(axis='index', how='all', inplace=True)
abs_data.index = abs_data.index.str.lower()

#load representatives data
representatives_data = pd.read_excel('./data/Representatives.xlsx', index_col='Electorate', usecols='C,F')
representatives_data.index = representatives_data.index.str.lower()

#merge both dataframes
abs_data = abs_data.merge(representatives_data, left_index=True, right_index=True)
abs_data['Climate Label'] = abs_data.apply(climate_label, axis=1)
columns = list(abs_data)

#create figure for the graphs
fig = plt.figure(figsize=[24, 13])

# determine layout of graphs
if (len(columns) - 2) <= 2 or (len(columns) - 2) > 4:
    fig_rows = int((len(columns) - 2)/5) + 1
    fig_cols = math.ceil((len(columns) - 2)/fig_rows)
else:
    fig_rows = 2
    fig_cols = 2
ax_number = 1

# create a graph per data column of selected dataset
for i in columns:  
    if i != 'Climate Change Support' and i != 'Climate Label':
        abs_subset = pd.DataFrame({i: abs_data[i], 'Climate Change Support': abs_data['Climate Change Support'], 'Climate Label': abs_data['Climate Label']}, index=abs_data.index) 
        
        #add new axes to figure
        ax = plt.subplot(fig_rows, fig_cols, ax_number)        
        ax.set_ylabel('Representatives/Electorates')

        #when using non-integer steps, results are inconsistent with using np.arange, so using linspace instead even though arange would be simpler to create the bins and labels from
        if abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][3] == 'steps':
            dataset_bin = np.linspace(start=abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][0], stop=abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][1], num=abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][2])
            dataset_bin = np.around(dataset_bin, decimals=2)
            dataset_labels = dataset_bin[1:]
        elif abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][3] == 'increment':
            number = round((abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][1] - abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][0])/abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][2]) + 1
            dataset_bin = np.linspace(start=abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][0], stop=abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][1], num=number)
            dataset_bin = np.around(dataset_bin, decimals=2)
            #dataset_labels = np.linspace(start=(abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][0] + abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][2]), stop=abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][1], num=(len(dataset_bin) - 1))
            dataset_labels = dataset_bin[1:]

        if abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][4] == 'int':
            #format as integers
            dataset_labels = ['{:,}'.format(round(x)) for x in dataset_labels]
            dataset_labels[0] = '<' + dataset_labels[0]
        elif abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][4] == 'perc':
            #format as percentage
            dataset_labels = [str(round(x * 100)) + '%' for x in dataset_labels]
            dataset_labels[0] = '<' + dataset_labels[0]
        elif abs_df_metadata.loc[abs_cat]['Bins'][ax_number - 1][4] == 'dollar':
            #format as dollar
            dataset_labels = ['$' + str(round(x)) for x in dataset_labels]
            dataset_labels[0] = '<' + dataset_labels[0]

        #add lines to axes
        if show_types_of_deniers:
            active_climate_deniers = abs_subset[abs_subset['Climate Change Support']==0]
            active_climate_deniers_bins = pd.cut(x=active_climate_deniers[i], bins=dataset_bin, labels=dataset_labels)
            active_climate_deniers_count = active_climate_deniers.groupby(active_climate_deniers_bins)[i].agg('count')
            active_climate_deniers_count.plot(ax=ax, color='m', label=active_climate_deniers['Climate Label'][0])

            climate_deniers = abs_subset[abs_subset['Climate Change Support']==1]
            climate_deniers_bins = pd.cut(x=climate_deniers[i], bins=dataset_bin, labels=dataset_labels)
            climate_deniers_count = climate_deniers.groupby(climate_deniers_bins)[i].agg('count')
            climate_deniers_count.plot(ax=ax, color='r', label=climate_deniers['Climate Label'][0])

            fence_sitters = abs_subset[abs_subset['Climate Change Support']==2]
            fence_sitters_bins = pd.cut(x=fence_sitters[i], bins=dataset_bin, labels=dataset_labels)
            fence_sitters_count = fence_sitters.groupby(fence_sitters_bins)[i].agg('count')
            fence_sitters_count.plot(ax=ax, color='y', label=fence_sitters['Climate Label'][0])

        if show_all_deniers:
            deniers_and_doubters = abs_subset[(abs_subset['Climate Change Support']>=0) & (abs_subset['Climate Change Support']<=2)]
            deniers_and_doubters_bins = pd.cut(x=deniers_and_doubters[i], bins=dataset_bin, labels=dataset_labels)
            deniers_and_doubters_count = deniers_and_doubters.groupby(deniers_and_doubters_bins)[i].agg('count')
            deniers_and_doubters_count.plot(ax=ax, color='b', label='Deniers and Doubters')

        if show_accepts_the_science:
            accepting_of_the_science = abs_subset[abs_subset['Climate Change Support']==3]
            accepting_of_the_science_bins = pd.cut(x=accepting_of_the_science[i], bins=dataset_bin, labels=dataset_labels)
            accepting_of_the_science_count = accepting_of_the_science.groupby(accepting_of_the_science_bins)[i].agg('count')
            accepting_of_the_science_count.plot(ax=ax, color='g', label=accepting_of_the_science['Climate Label'][0])

        if show_total:
            show_total_bins = pd.cut(x=abs_subset[i], bins=dataset_bin, labels=dataset_labels)
            show_total_count = abs_subset.groupby(show_total_bins)[i].agg('count')
            show_total_count.plot(ax=ax, color='k', label='Total ' + abs_cat)

        ax.grid(True)
        ax.legend()
        ax_number += 1

plt.suptitle('Federal Climate Change Denial Electoral Representation by ' + abs_cat)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
#plt.show()
plt.savefig('Representation by ' + abs_cat + '.png')
