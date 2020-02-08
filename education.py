import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


electorates = pd.read_csv('./data/education.csv')

active_climate_deniers = 0
climate_deniers = 1
fence_sitters = 2
accepts_the_science = 3

representative_active_climate_deniers = electorates[electorates['Denialists']==active_climate_deniers]
representative_climate_deniers = electorates[electorates['Denialists']==climate_deniers]
representative_fence_sitters = electorates[electorates['Denialists']==fence_sitters]
representative_accepting_of_the_science = electorates[electorates['Denialists']==accepts_the_science]

colors = ['m', 'r', 'y', 'g']
names = ['Active Denialists', 'Climate Denialists', 'Inbetweeners', 'Accepts the Science']
kwargs = dict(alpha=0.5, bins=20, stacked=True)

#plt.hist([representative_active_climate_deniers['Year 12 or equivalent completion (people aged 20 to 24 years)'], representative_climate_deniers['Year 12 or equivalent completion (people aged 20 to 24 years)'], representative_fence_sitters['Year 12 or equivalent completion (people aged 20 to 24 years)'], representative_accepting_of_the_science['Year 12 or equivalent completion (people aged 20 to 24 years)']], bins = 20, color = colors, label = names)
plt.hist(representative_active_climate_deniers['Year 12 or equivalent completion (people aged 20 to 24 years)'], **kwargs, color=colors[active_climate_deniers], label=names[active_climate_deniers])
plt.hist(representative_climate_deniers['Year 12 or equivalent completion (people aged 20 to 24 years)'], **kwargs, color=colors[climate_deniers], label=names[climate_deniers])
plt.hist(representative_fence_sitters['Year 12 or equivalent completion (people aged 20 to 24 years)'], **kwargs, color=colors[fence_sitters], label=names[fence_sitters])
plt.hist(representative_accepting_of_the_science['Year 12 or equivalent completion (people aged 20 to 24 years)'], **kwargs, color=colors[accepts_the_science], label=names[accepts_the_science])

plt.legend()
plt.xlabel('Year 12 Completion Rate')
plt.ylabel('Electorates')
plt.title('Climate Denialists by Year 12 completion rates of electorates')
#plt.show()
plt.savefig('./diagrams/histograms/Year12 completion rates and climate denial representation.png')
