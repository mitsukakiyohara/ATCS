__version__ = '3.1'
__author__ = 'Mitsuka Kiyohara'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sfcrime = pd.read_csv("SFCrime__2018_to_Present.csv")
historical_sfcrime = pd.read_csv("SFCrime_2003_to_May_2018.csv")

#sfcrime.info()
#historical_sfcrime.info()

#--basic info--

#Under disorderly conduct, 491 incidents are from public drunkenness 
#71% of people are arrested for public intoxication , 30% of people are open or active 
#98 cases in Mission, 46 in Tenderloin, and 45 in Soma (neighborhood), 130 in Mission, 82 in Central, 73 in Southern
print(sfcrime[ sfcrime['Incident Subcategory'] == 'Drunkenness']['Analysis Neighborhood'].value_counts())

#(2003-2018) 2155 in Southern, 1806 in Mission, Central in 1203
#total cases: 9826, ranking #22

print(historical_sfcrime[ historical_sfcrime['Category'] == 'DRUNKENNESS']['PdDistrict'].value_counts())



#--graphs--

#only reports within SF
sfcrime = sfcrime [ sfcrime['Police District'] != 'Out of SF' ]
#only for 2018
sfcrime = sfcrime [ sfcrime['Incident Year'] == 2018 ]
#only for categories related to drunkenness
sfcrime = sfcrime [ sfcrime['Incident Subcategory'] == 'Drunkenness']


crime_distcat = pd.crosstab(index=sfcrime['Police District'],
                            columns=sfcrime['Incident Subcategory'])
crime_distcat['Total'] = crime_distcat.apply(sum, axis=1)
sfdistricts = pd.read_csv("SF_Police_Districts.csv", index_col='PdDistrict')
sfcrime_districts = pd.concat( [crime_distcat, sfdistricts], axis=1, sort=False)


sfcrime_districts['per_area']=sfcrime_districts['Total']/sfcrime_districts['Land Mass']

plotme = sfcrime_districts.sort_values('per_area',ascending=False)
plotme.plot(y='per_area',kind='bar')

#plot for per-area
plt.title('Public Drunkenness Incidents per Area in Each District for 2018')
plt.xlabel("District")
plt.ylabel("# of Incidents")
#plt.show()


#plot for tenderlion (# of drunk incidents by day of week)
tenderloin = sfcrime[sfcrime['Analysis Neighborhood'] == 'Tenderloin']
data=tenderloin[ tenderloin['Incident Subcategory'] == 'Drunkenness']['Incident Day of Week'].value_counts()
data.plot(kind='bar')
plt.axis(ymin=0, ymax=10)
plt.ylabel("# of Drunk Incidents")
plt.xlabel("Day of the Week")
plt.title('Number of Drunk Incidents in Tenderloin by Day of the Week in 2018')
#plt.show()

#plot for chinatown (# of drunk incidents by day of week)
chinatown = sfcrime[sfcrime['Analysis Neighborhood'] == 'Chinatown']
data=chinatown[ chinatown['Incident Subcategory'] == 'Drunkenness']['Incident Day of Week'].value_counts()
data.plot(kind='bar')
plt.axis(ymin=0, ymax=5)
plt.ylabel("# of Drunk Incidents")
plt.xlabel("Day of the Week")
plt.title('Number of Drunk Incidents in Chinatown by Day of the Week in 2018')
#plt.show()

#plot for nob hill (# of drunk incidents by day of week)
nobhill = sfcrime[sfcrime['Analysis Neighborhood'] == 'Nob Hill']
data=nobhill[ nobhill['Incident Subcategory'] == 'Drunkenness']['Incident Day of Week'].value_counts()
data.plot(kind='bar')
plt.axis(ymin=0, ymax=5)
plt.ylabel("# of Drunk Incidents")
plt.xlabel("Day of the Week")
plt.title('Number of Drunk Incidents in Nob Hill by Day of the Week in 2018')
#plt.show()

#plot for financial district (# of drunk incidents by day of week)
financial = sfcrime[sfcrime['Analysis Neighborhood'] == 'Financial District/South Beach']
data=financial[ financial['Incident Subcategory'] == 'Drunkenness']['Incident Day of Week'].value_counts()
data.plot(kind='bar')
plt.axis(ymin=0, ymax=10)
plt.ylabel("# of Drunk Incidents")
plt.xlabel("Day of the Week")
plt.title('Number of Drunk Incidents in Financial District by Day of the Week in 2018')
#plt.show()

#location map of public drunk incidents for 2018
drunk = sfcrime[ sfcrime['Incident Subcategory'] == 'Drunkenness']
plt.scatter(x=drunk['Latitude'], y=drunk['Longitude'],alpha=0.3)
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Publicly Drunk Incidents by Location in SF for 2018")
#plt.show()

#location map of public drunk incidents for 2003-2018
historical_drunk = historical_sfcrime[ historical_sfcrime['Category'] == 'DRUNKENNESS']
plt.scatter(x=historical_drunk['Y'], y=historical_drunk['X'],alpha=0.3)
plt.title("Publicly Drunk Incidents by Location in SF from 2003-2018")
#plt.show()


#define color to each incident description
#public drunk in incidents for 2003-2018 (with color markers)
def res_to_color (res):
	res_colors = { 'ARREST, BOOKED' : 'red',
					'NONE' : 'green', 
					'ARREST, CITED' : 'yellow',
					'JUVENILE BOOKED' : 'blue' }
	if res in res_colors:
		return res_colors[res]
	else:
		return 'grey'

historical_drunk_us = historical_sfcrime[ historical_sfcrime['Category'] == 'DRUNKENNESS']
plt.scatter(x=historical_drunk_us['X'], y=historical_drunk_us['Y'], c=historical_drunk_us['Resolution'].apply(res_to_color), alpha=0.3)
plt.title("Publicly Drunk Incidents in San Francisco from 2003-2018")
#plt.show()


#trendline/line graph for number of drunk incident based on time  
drunk = historical_sfcrime [ historical_sfcrime['Category'] == 'DRUNKENNESS']
drunk_crime=drunk.groupby('Time').aggregate(sum)

drunk_crime.plot(kind='line')
plt.ylabel('Number of Drunk Incidents')
plt.title('Number of Drunk Incidents Based on Time (2003-2018)')
plt.show()






