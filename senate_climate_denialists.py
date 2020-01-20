"""Maps Federal Representatives position on climate change based on their voting history to the jurisdications on an Australian map

This product (climate_denialists.py) incorporates data that is:
    © Commonwealth of Australia (Australian Electoral Commission) 2020
    © Commonwealth of Australia (Australian Bureau of Statistics) 2020

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapefile as shp
#hack to get Basemap working in default install environment
import os
os.environ['PROJ_LIB']='C:\\Users\\Andy\\Anaconda3\\pkgs\\proj4-5.2.0-ha925a31_1\\Library\\share'
from mpl_toolkits.basemap import Basemap


__author__ = "Andrew Arch"
__copyright__ = "Copyright 2020"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Andrew Arch"
__email__ = "andy.arch11@gmail.com"
__status__ = "Production"

#create figure to host map.  N.B. if this is declared after Basemap, it will plot points to its own window
fig = plt.figure(figsize=[24, 13], constrained_layout=True)
ax = fig.add_subplot()

#turn off any axes ticks on the diagram
plt.xticks(visible=False)
plt.yticks(visible=False)

plt.title('Climate Denial in the Australian Senate')

#TODO address the deprecation warning coming from the Basemap constructor call - The dedent function was deprecated in Matplotlib 3.1 and will be removed in 3.3. Use inspect.cleandoc instead.
#draw map - Australia -26.8372557,127.5859928
m = Basemap(projection='aea', lat_0=-22.5, lon_0=130.5, resolution='l', llcrnrlat=-42, urcrnrlat=-3, llcrnrlon=95, urcrnrlon=166, ax=ax)
m.drawcoastlines(color='0', linewidth=0.25)
m.fillcontinents(color='white', alpha=1)
m.drawparallels(np.arange(-47,0,23.5))
m.drawmeridians(np.arange(100,180,20))

m.readshapefile('./data/Igismap/Australia_Polygon', 'Australia_Polygon')

def fill_cell(x, y, colour, value):
    #fill given cell.
    alpha = 0.6
    if int(value) > 0:
        alpha = 1

    ax.fill(x, y, color=colour, transform=ax.transAxes, alpha=alpha, zorder=10)

def display_table(cell_values, upper_left_corner, state):
    #print the summary table
    # senator_box_ulc = [0.005, 0.845]    
    # senator_box_lrc = [0.301, 0.802]
    senator_box_ulc = [upper_left_corner[0], upper_left_corner[1]]
    senator_box_lrc = [senator_box_ulc[0] + 0.28, senator_box_ulc[1] - 0.07]
    senator_box_llc = [senator_box_ulc[0], senator_box_lrc[1]]
    senator_box_urc = [senator_box_lrc[0], senator_box_ulc[1]]
    senator_box_ml1 = [senator_box_ulc[0], senator_box_ulc[1] - ((senator_box_ulc[1] - senator_box_llc[1])/4)]
    senator_box_mr1 = [senator_box_urc[0], senator_box_ml1[1]]
    senator_box_ml2 = [senator_box_ulc[0], senator_box_llc[1] + ((senator_box_ulc[1] - senator_box_llc[1])/2)]
    senator_box_mr2 = [senator_box_urc[0], senator_box_ml2[1]]
    senator_box_ml3 = [senator_box_ulc[0], senator_box_llc[1] + ((senator_box_ulc[1] - senator_box_llc[1])/4)]
    senator_box_mr3 = [senator_box_urc[0], senator_box_ml3[1]]
    senator_box_vert1_x = upper_left_corner[0] + 0.061
    senator_box_vert2_x = senator_box_vert1_x + 0.052
    senator_box_vert3_x = senator_box_vert2_x + 0.057
    senator_box_vert4_x = senator_box_vert3_x + 0.0465
    
    senator_heading_y = senator_box_ulc[1] - 0.015
    senator_term1_values_y = senator_box_mr1[1] - 0.015
    senator_term2_values_y = senator_box_mr2[1] - 0.015
    senator_total_values_y = senator_box_mr3[1] - 0.015

    type_txt_x = senator_box_ulc[0] + 0.005
    active_deniers_txt_x = senator_box_vert1_x + 0.005
    deniers_txt_x = senator_box_vert2_x + 0.005
    inbetweeners_txt_x = senator_box_vert3_x + 0.005
    accepts_science_txt_x = senator_box_vert4_x + 0.005

    #fill first column
    senator_box_fill_col0_x = [senator_box_ulc[0], senator_box_vert1_x, senator_box_vert1_x, senator_box_ulc[0], senator_box_ulc[0]]
    senator_box_fill_y = [senator_box_ulc[1], senator_box_ulc[1], senator_box_llc[1], senator_box_llc[1], senator_box_ulc[1]]
    ax.fill(senator_box_fill_col0_x, senator_box_fill_y, color='w', transform=ax.transAxes, zorder=10)

    #fill heading row
    senator_box_fill_col1_x = [senator_box_vert1_x, senator_box_vert2_x, senator_box_vert2_x, senator_box_vert1_x, senator_box_vert1_x]
    senator_box_fill_row0_y = [senator_box_ulc[1], senator_box_ulc[1], senator_box_ml1[1], senator_box_ml1[1], senator_box_ulc[1]]
    ax.fill(senator_box_fill_col1_x, senator_box_fill_row0_y, color='m', transform=ax.transAxes, zorder=10)
    
    senator_box_fill_col2_x = [senator_box_vert2_x, senator_box_vert3_x, senator_box_vert3_x, senator_box_vert2_x, senator_box_vert2_x]
    ax.fill(senator_box_fill_col2_x, senator_box_fill_row0_y, color='r', transform=ax.transAxes, zorder=10)

    senator_box_fill_col3_x = [senator_box_vert3_x, senator_box_vert4_x, senator_box_vert4_x, senator_box_vert3_x, senator_box_vert3_x]
    ax.fill(senator_box_fill_col3_x, senator_box_fill_row0_y, color='y', transform=ax.transAxes, zorder=10)
    
    senator_box_fill_col4_x = [senator_box_vert4_x, senator_box_urc[0], senator_box_lrc[0], senator_box_vert4_x, senator_box_vert4_x]
    ax.fill(senator_box_fill_col4_x, senator_box_fill_row0_y, color='g', transform=ax.transAxes, zorder=10)

    line_colour = 'k'
    #outline
    senator_box_x = [senator_box_ulc[0], senator_box_urc[0], senator_box_lrc[0], senator_box_llc[0], senator_box_ulc[0]]
    senator_box_y = [senator_box_ulc[1], senator_box_urc[1], senator_box_lrc[1], senator_box_llc[1], senator_box_ulc[1]]
    ax.plot(senator_box_x, senator_box_y, color=line_colour, linewidth=0.25, transform=ax.transAxes, zorder=15)

    #midline
    senator_box_x = [senator_box_ml1[0], senator_box_mr1[0]]
    senator_box_y = [senator_box_ml1[1], senator_box_mr1[1]]
    ax.plot(senator_box_x, senator_box_y, color=line_colour, linewidth=0.25, transform=ax.transAxes, zorder=15)

    senator_box_x = [senator_box_ml2[0], senator_box_mr2[0]]
    senator_box_y = [senator_box_ml2[1], senator_box_mr2[1]]
    ax.plot(senator_box_x, senator_box_y, color=line_colour, linewidth=0.25, transform=ax.transAxes, zorder=15)

    senator_box_x = [senator_box_ml3[0], senator_box_mr3[0]]
    senator_box_y = [senator_box_ml3[1], senator_box_mr3[1]]
    ax.plot(senator_box_x, senator_box_y, color=line_colour, linewidth=0.25, transform=ax.transAxes, zorder=15)

    #vertical divider
    senator_box_x = [senator_box_vert1_x, senator_box_vert1_x]
    senator_box_y = [senator_box_ulc[1], senator_box_llc[1]]
    ax.plot(senator_box_x, senator_box_y, color=line_colour, linewidth=0.25, transform=ax.transAxes, zorder=15)

    #vertical_divider
    senator_box_x = [senator_box_vert2_x, senator_box_vert2_x]
    senator_box_y = [senator_box_ulc[1], senator_box_llc[1]]
    ax.plot(senator_box_x, senator_box_y, color=line_colour, linewidth=0.25, transform=ax.transAxes, zorder=15)
    
    #vertical_divider
    senator_box_x = [senator_box_vert3_x, senator_box_vert3_x]
    senator_box_y = [senator_box_ulc[1], senator_box_llc[1]]
    ax.plot(senator_box_x, senator_box_y, color=line_colour, linewidth=0.25, transform=ax.transAxes, zorder=15)
    
    #vertical_divider
    senator_box_x = [senator_box_vert4_x, senator_box_vert4_x]
    senator_box_y = [senator_box_ulc[1], senator_box_llc[1]]
    ax.plot(senator_box_x, senator_box_y, color=line_colour, linewidth=0.25, transform=ax.transAxes, zorder=15)

    row1_box_y = [senator_box_ml1[1], senator_box_ml1[1], senator_box_ml2[1], senator_box_ml2[1], senator_box_ml1[1]]
    row2_box_y = [senator_box_ml2[1], senator_box_ml2[1], senator_box_ml3[1], senator_box_ml3[1], senator_box_ml2[1]]
    row3_box_y = [senator_box_ml3[1], senator_box_ml3[1], senator_box_lrc[1], senator_box_lrc[1], senator_box_ml3[1]]
    
    #header row text
    ax.text(type_txt_x,senator_heading_y, state, horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    ax.text(active_deniers_txt_x,senator_heading_y, 'Active Deniers', horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    ax.text(deniers_txt_x, senator_heading_y, 'Climate Deniers', horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    ax.text(inbetweeners_txt_x, senator_heading_y, 'Inbetweeners', horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    ax.text(accepts_science_txt_x, senator_heading_y, 'Accepts the Science', horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)   
    #2022 Term
    ax.text(type_txt_x, senator_term1_values_y, 'Senators (' + str(first_term) + '): ', horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col1_x, row1_box_y, 'm', cell_values[0][0])
    ax.text(active_deniers_txt_x, senator_term1_values_y, cell_values[0][0], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col2_x, row1_box_y, 'r', cell_values[0][1])
    ax.text(deniers_txt_x, senator_term1_values_y, cell_values[0][1], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col3_x, row1_box_y, 'y', cell_values[0][2])
    ax.text(inbetweeners_txt_x, senator_term1_values_y, cell_values[0][2], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col4_x, row1_box_y, 'g', cell_values[0][3])
    ax.text(accepts_science_txt_x, senator_term1_values_y, cell_values[0][3], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    #2025 Term       
    ax.text(type_txt_x, senator_term2_values_y, 'Senators (' + str(second_term) + '): ', horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col1_x, row2_box_y, 'm', cell_values[1][0])
    ax.text(active_deniers_txt_x, senator_term2_values_y, cell_values[1][0], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col2_x, row2_box_y, 'r', cell_values[1][1])
    ax.text(deniers_txt_x, senator_term2_values_y, cell_values[1][1], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col3_x, row2_box_y, 'y', cell_values[1][2])
    ax.text(inbetweeners_txt_x, senator_term2_values_y, cell_values[1][2], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col4_x, row2_box_y, 'g', cell_values[1][3])
    ax.text(accepts_science_txt_x, senator_term2_values_y, cell_values[1][3], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    #Total       
    ax.text(type_txt_x, senator_total_values_y, 'Senators (total): ', horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col1_x, row3_box_y, 'm', cell_values[2][0])
    ax.text(active_deniers_txt_x, senator_total_values_y, cell_values[2][0], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col2_x, row3_box_y, 'r', cell_values[2][1])
    ax.text(deniers_txt_x, senator_total_values_y, cell_values[2][1], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col3_x, row3_box_y, 'y', cell_values[2][2])
    ax.text(inbetweeners_txt_x, senator_total_values_y, cell_values[2][2], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)
    fill_cell(senator_box_fill_col4_x, row3_box_y, 'g', cell_values[2][3])
    ax.text(accepts_science_txt_x, senator_total_values_y, cell_values[2][3], horizontalalignment='left', color='k', transform=ax.transAxes, fontsize=9, zorder=20)

state_names = pd.read_csv('./data/1989 States.csv')

senators = pd.read_excel('./data/Senate.xlsx', index_col='Senator')
senator_active_climate_deniers = senators[senators['Climate Change Support']==0]
senator_climate_deniers = senators[senators['Climate Change Support']==1]
senator_fence_sitters = senators[senators['Climate Change Support']==2]
senator_accepting_of_the_science = senators[senators['Climate Change Support']==3]

#columns = ('Active Deniers', 'Climate Deniers', 'Inbetweeners', 'Accepts the Science')
#rows = ('Term 2022', 'Term 2025', 'Total')
first_term = 2022
second_term = 2025

cell_text = []
cell_row = []
term22_senator_active_climate_deniers = senators[(senators['Climate Change Support']==0) & (senators['End Term']==first_term)]
cell_row.append(str(len(term22_senator_active_climate_deniers)))
term22_senator_climate_deniers = senators[(senators['Climate Change Support']==1) & (senators['End Term']==first_term)]
cell_row.append(str(len(term22_senator_climate_deniers)))
term22_senator_fence_sitters = senators[(senators['Climate Change Support']==2) & (senators['End Term']==first_term)]
cell_row.append(str(len(term22_senator_fence_sitters)))
term22_senator_accepting_of_the_science = senators[(senators['Climate Change Support']==3) & (senators['End Term']==first_term)]
cell_row.append(str(len(term22_senator_accepting_of_the_science)))
cell_text.append(cell_row)

cell_row = []
term25_senator_active_climate_deniers = senators[(senators['Climate Change Support']==0) & (senators['End Term']==second_term)]
cell_row.append(str(len(term25_senator_active_climate_deniers)))
term25_senator_climate_deniers = senators[(senators['Climate Change Support']==1) & (senators['End Term']==second_term)]
cell_row.append(str(len(term25_senator_climate_deniers)))
term25_senator_fence_sitters = senators[(senators['Climate Change Support']==2) & (senators['End Term']==second_term)]
cell_row.append(str(len(term25_senator_fence_sitters)))
term25_senator_accepting_of_the_science = senators[(senators['Climate Change Support']==3) & (senators['End Term']==second_term)]
cell_row.append(str(len(term25_senator_accepting_of_the_science)))
cell_text.append(cell_row)

cell_row = []
senator_active_climate_deniers = senators[(senators['Climate Change Support']==0)]
cell_row.append(str(len(senator_active_climate_deniers)))
senator_climate_deniers = senators[(senators['Climate Change Support']==1)]
cell_row.append(str(len(senator_climate_deniers)))
senator_fence_sitters = senators[(senators['Climate Change Support']==2)]
cell_row.append(str(len(senator_fence_sitters)))
senator_accepting_of_the_science = senators[(senators['Climate Change Support']==3)]
cell_row.append(str(len(senator_accepting_of_the_science)))
cell_text.append(cell_row)

display_table(cell_text, (0.005, 0.845), 'AU Senate')

for index, state_name in state_names.iterrows(): 
    #x0,y0 = m(float(state_name['Longitude']), float(state_name['Latitude']))
    statename = state_name['StateName']
    x0 = float(state_name['Longitude'])
    y0 = float(state_name['Latitude'])
    #ax.text(x0, y0, state_name['StateName'], verticalalignment='center', horizontalalignment='left', color='0.6', fontsize=9)
        
    cell_text = []
    cell_row = []
    state_term22_senator_active_climate_deniers = senators[(senators['Climate Change Support']==0) & (senators['State']==state_name['StateName']) & (senators['End Term']==first_term)]
    cell_row.append(str(len(state_term22_senator_active_climate_deniers)))
    state_term22_senator_climate_deniers = senators[(senators['Climate Change Support']==1) & (senators['State']==state_name['StateName']) & (senators['End Term']==first_term)]
    cell_row.append(str(len(state_term22_senator_climate_deniers)))
    state_term22_senator_fence_sitters = senators[(senators['Climate Change Support']==2) & (senators['State']==state_name['StateName']) & (senators['End Term']==first_term)]
    cell_row.append(str(len(state_term22_senator_fence_sitters)))
    state_term22_senator_accepting_of_the_science = senators[(senators['Climate Change Support']==3) & (senators['State']==state_name['StateName']) & (senators['End Term']==first_term)]
    cell_row.append(str(len(state_term22_senator_accepting_of_the_science)))
    cell_text.append(cell_row)
    
    cell_row = []
    state_term25_senator_active_climate_deniers = senators[(senators['Climate Change Support']==0) & (senators['State']==state_name['StateName']) & (senators['End Term']==second_term)]
    cell_row.append(str(len(state_term25_senator_active_climate_deniers)))
    state_term25_senator_climate_deniers = senators[(senators['Climate Change Support']==1) & (senators['State']==state_name['StateName']) & (senators['End Term']==second_term)]
    cell_row.append(str(len(state_term25_senator_climate_deniers)))
    state_term25_senator_fence_sitters = senators[(senators['Climate Change Support']==2) & (senators['State']==state_name['StateName']) & (senators['End Term']==second_term)]
    cell_row.append(str(len(state_term25_senator_fence_sitters)))
    state_term25_senator_accepting_of_the_science = senators[(senators['Climate Change Support']==3) & (senators['State']==state_name['StateName']) & (senators['End Term']==second_term)]
    cell_row.append(str(len(state_term25_senator_accepting_of_the_science)))
    cell_text.append(cell_row)

    cell_row = []
    state_senator_active_climate_deniers = senators[(senators['Climate Change Support']==0) & (senators['State']==state_name['StateName'])]
    cell_row.append(str(len(state_senator_active_climate_deniers)))
    state_senator_climate_deniers = senators[(senators['Climate Change Support']==1) & (senators['State']==state_name['StateName'])]
    cell_row.append(str(len(state_senator_climate_deniers)))
    state_senator_fence_sitters = senators[(senators['Climate Change Support']==2) & (senators['State']==state_name['StateName'])]
    cell_row.append(str(len(state_senator_fence_sitters)))
    state_senator_accepting_of_the_science = senators[(senators['Climate Change Support']==3) & (senators['State']==state_name['StateName'])]
    cell_row.append(str(len(state_senator_accepting_of_the_science)))
    cell_text.append(cell_row)

    #plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, bbox=[x0, y0, x0 + 0.12, y0 + 0.01], zorder=50)
    #plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc=loc)
    if statename == 'Australian Capital Territory':
        statename = 'ACT'
    display_table(cell_text, (x0, y0), statename)

#display the map
#plt.show()
plt.savefig('Australian_Senate_Climate_Denialists.png')