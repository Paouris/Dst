#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dst estimates:
    
-Input:
    1-minute (or whatever) Solar Orbiter data:
        Density
        Speed
        Magnetic Field B_N (RTN)
-Output:
    Dst estimates using the 
    a) Burton et al., 1975 and 
    b) O'Brien and McPherron, 2000 models
    
-Note: According the time resolution of the input (1-minute or 1-hour) we have to make 
changes to the time scale for Dst lines 167 and 196.

-The constants and the equations are from:
    (a) Burton et al., 1975 paper,
    (b) O'Brien and McPherron 2000 paper.    

-Validation of the code/procedure: 
    We used as input the data from STEREO-A and we reproduced
    the Dst estimates exactly as they presented in Figure 5 / bottom panel
    at Liu et al., 2014 paper, DOI: 10.1038/ncomms4481.

-This code used for the calculations of Dst index utilizing Solar Orbiter's 
insitu data presented in the paper: "The Space Weather Context of the first 
extreme event of Solar Cycle 25, on September 5,  2022" at ApJ, Paouris et al., 2023.

@author: paoure1
"""


#Import the libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import copy

#Read and Store the data into a data frame:
#df = pd.read_csv('/Users/paoure1/Projects2/event_list/20220905/Dst_create/dst_v2.csv', skiprows=1, names=['Year','DOY','Hour','Scalar_B','B_z','Temperature','Density','Speed','Electric_Field','Dst_obs'])
oFilePath = '/Users/paoure1/Projects2/event_list/20220905_all/SOLO_data/'
df = pd.read_csv(oFilePath + 'new_1min_data.csv', skiprows=1, names=["epoch", "B_R", "B_T", "B_N", "B_mag", "V_R", "V_T", "V_N", "V_mag", "Density"])
#df = pd.read_csv(oFilePath + '20min_data.csv', skiprows=0, names=["epoch", "B_N", "V_mag", "Density"])
#df = pd.read_csv(oFilePath + 'SOLO_for_Dst.csv', skiprows=0, names=["epoch", "B_R", "B_T", "B_N", "B_mag", "V_R", "V_T", "V_N", "V_mag", "Density"])

#Check:
print(df.head(6))
print(df.info())


date_time = df["epoch"]
print(date_time[0:2])
date_time = pd.to_datetime(date_time)

#Check:
var = 'B_N'

#---
# Make arrays:

# Density:
density = df['Density']
density = np.array(density)
density_1au = density * 0.4248

# Speed:
speed = df['V_mag']
speed = np.array(speed)

# =============================================================================
# # Temperature:
# Temperature = df['Temperature']
# Temperature = np.array(Temperature)
# =============================================================================

# Bz:
Bz = df[var]
Bz = np.array(Bz)
print(Bz[0:5])
Bz_1au = Bz * 0.5
#--- all components for plot:
BzT = df['B_T']
BzT = np.array(BzT)
print(BzT[0:5])
BzR = df['B_R']
BzR = np.array(BzR)
print(BzR[0:5])

BzN = df['B_N']
BzN = np.array(BzN)
print(BzN[0:5])
BzN_1au = BzN*0.5
#---

size_arr = len(Bz)
print(size_arr)



#Create the empty arrays (zeros):
Ey = np.zeros(size_arr)
Dst_star_Burton = np.zeros(size_arr)
Dst_Burton = np.zeros(size_arr)
Dst_Burton_1au = np.zeros(size_arr)
Dynamic_Pressure = np.zeros(size_arr)
Dynamic_Pressure_1au = np.zeros(size_arr)
Dst_star_OBrien_McPherron = np.zeros(size_arr)
Dst_OBrien_McPherron = np.zeros(size_arr)
Dst_OBrien_McPherron_1au = np.zeros(size_arr)
#---


Bz_positive = np.where(Bz > 0)
Bz_pos = copy.deepcopy(Bz)
Bz_pos[Bz_positive] = 0
print(Bz_pos[0:5])

Ey = speed * abs(Bz_pos) * 1.0E-03 #in mV/meter
print(Ey[0:5])



protonmass = 1.67262192*1.0E-27 # in kilograms
#Dynamic Pressure:
# P_dy = m_p * N_p * V_p^2
#where  m_p = proton mass, (in kilograms)
#       N_p = density, and (in 1/cm^3)
#       V_p = speed (in km/s)        
Dynamic_Pressure = (density * 1.0E+06 * protonmass * (speed * 1.0E+03)**2.0)*1.0E+09 # in nPa

# comments for the units:
# density * 1.0E+06 = now in m^3
# speed * 1.0E+03 = now in m/s (instead of km/s)
# *1.0E+09 = now in nanoPascal (nPa)

# Create the time series:
time_series = np.arange(0,size_arr,1)
#print(time_series)



# =============================================================================
# Burton et al., 1975 - Dst estimations:
# =============================================================================

# Define the constants - see Burton paper 1975, p. 4208:
# 1 gamma = 1 nT
Ec = 0.5 #in mV/meter
a  = 3.60*1E-05 #in sec^(-1)
b  = 0.20/(np.sqrt(1.6021766E-04)) #in nT / ( (nPa)^0.5 )
c  = 20.0 #in nT
d  = -1.50*1E-03 #in nT * (mV/meter)^-1 * sec^-1

for i in range(size_arr-1):
  # print(Ey[i])
  if Ey[i] > Ec:
    #Ring current injection F(Ey) in terms of electric field Ey:
    F = d * (Ey[i] - Ec)
  else: F = 0
  # print(F)
  #deltat_sec = (time_series[i+1] - time_series[i]) * 3600.0 #1 hour data = 3600.0 seconds
  deltat_sec = (time_series[i+1] - time_series[i]) * 60.0 #1 min data = 60.0 seconds
  # print(' ')
  # print('time1: ', time_series[i+1])
  # print('time2: ', time_series[i])
  Dst_star_Burton[i+1] = ( F - a * Dst_star_Burton[i] ) * deltat_sec + Dst_star_Burton[i]
  Dst_Burton[i+1] = Dst_star_Burton[i+1] + b * np.sqrt(Dynamic_Pressure[i+1]) - c
#print(Dst_Burton[0:5])

# =============================================================================
# 1au
# =============================================================================
Dynamic_Pressure_1au = (density_1au * 1.0E+06 * protonmass * (speed * 1.0E+03)**2.0)*1.0E+09

Bz_positive_1au = np.where(Bz_1au > 0)
Bz_pos_1au = copy.deepcopy(Bz_1au)
Bz_pos_1au[Bz_positive_1au] = 0
print("Bz - 1au", Bz_pos_1au[0:5])

Ey_1au = speed * abs(Bz_pos_1au) * 1.0E-03 #in mV/meter
print(Ey_1au[0:5])
#---1au
for i in range(size_arr-1):
  # print(Ey[i])
  if Ey_1au[i] > Ec:
    #Ring current injection F(Ey) in terms of electric field Ey:
    F = d * (Ey_1au[i] - Ec)
  else: F = 0
  # print(F)
  #deltat_sec = (time_series[i+1] - time_series[i]) * 3600.0 #1 hour data = 3600.0 seconds
  deltat_sec = (time_series[i+1] - time_series[i]) * 60.0 #1 min data = 60.0 seconds
  # print(' ')
  # print('time1: ', time_series[i+1])
  # print('time2: ', time_series[i])
  Dst_star_Burton[i+1] = ( F - a * Dst_star_Burton[i] ) * deltat_sec + Dst_star_Burton[i]
  Dst_Burton_1au[i+1] = Dst_star_Burton[i+1] + b * np.sqrt(Dynamic_Pressure_1au[i+1]) - c
#print(Dst_Burton_1au[0:5])
# =============================================================================
# 
# =============================================================================
print(" ")
print("========= Burton et al., 1975 model - 1 minute Dst =========")
print("Burton et al., 1975 Dst: ")
print("min Dst: ", "%4.1f" % min(Dst_Burton))
print("max Dst: ", "%4.1f" % max(Dst_Burton))
print("+++ extrapolate to 1au: +++")
print("min Dst: ", "%4.1f" % min(Dst_Burton_1au))
print("max Dst: ", "%4.1f" % max(Dst_Burton_1au))
print("==================")
print(" ")
# =============================================================================
# =============================================================================


# =============================================================================
# O'Brien and Mc Pherron 2000 Dst:
# =============================================================================

# Define the constants - see O'Brien's and McPherron's paper 2000:
Ec = 0.49 #in mV/meter
b  = 7.26 #in nT * (nPa)^-0.5
c  = 11.0  #in nT
for i in range(size_arr-1):
  if Ey[i] > Ec:
    #Ring current injection Q:
    Q=-4.4*(Ey[i]-Ec) 
  else: Q=0
  Decay_time = 2.4*np.exp(9.74/(4.69+Ey[i])) #Decay_time in hours according to O'Brien and McPherron paper.
  #Calculate Dst [1]
  #deltat_hours=(time_series[i+1]-time_series[i])*1.0 #time_series is in days - convert to hours: 1.0 is for 1 hour data
  deltat_hours=(time_series[i+1]-time_series[i])*(1./60.) #time_series is in days - convert to hours, 20-min, 1-min ...
  Dst_star_OBrien_McPherron[i+1]=((Q-Dst_star_OBrien_McPherron[i]/Decay_time))*deltat_hours+Dst_star_OBrien_McPherron[i]
  #Calculate Dst [2] - This is the Dst estimates: 
  Dst_OBrien_McPherron[i+1]=Dst_star_OBrien_McPherron[i+1]+b*np.sqrt(Dynamic_Pressure[i+1])-c
#print(Dst_OBrien_McPherron[0:5])
# =============================================================================
# 1au
# =============================================================================
for i in range(size_arr-1):
  if Ey_1au[i] > Ec:
    #Ring current injection Q:
    Q=-4.4*(Ey_1au[i]-Ec) 
  else: Q=0
  Decay_time = 2.4*np.exp(9.74/(4.69+Ey_1au[i])) #Decay_time in hours according to O'Brien and McPherron paper.
  #Calculate Dst [1]
  #deltat_hours=(time_series[i+1]-time_series[i])*1.0 #time_series is in days - convert to hours: 1.0 is for 1 hour data
  deltat_hours=(time_series[i+1]-time_series[i])*(1./60.) #time_series is in days - convert to hours, 20-min, 1-min ...
  Dst_star_OBrien_McPherron[i+1]=((Q-Dst_star_OBrien_McPherron[i]/Decay_time))*deltat_hours+Dst_star_OBrien_McPherron[i]
  #Calculate Dst [2] - This is the Dst estimates:
  Dst_OBrien_McPherron_1au[i+1]=Dst_star_OBrien_McPherron[i+1]+b*np.sqrt(Dynamic_Pressure_1au[i+1])-c
#print(Dst_OBrien_McPherron_1au[0:5])
# =============================================================================
# 
# =============================================================================
print("===== O'Brien and McPherron, 2000 model - 1 minute Dst =====")
print("O'Brien and McPherron, 2000 Dst:")
print("min Dst: ", "%4.1f" % min(Dst_OBrien_McPherron))
print("max Dst: ", "%4.1f" % max(Dst_OBrien_McPherron))
print("+++ extrapolate to 1au: +++")
print("min Dst: ", "%4.1f" % min(Dst_OBrien_McPherron_1au))
print("max Dst: ", "%4.1f" % max(Dst_OBrien_McPherron_1au))
print("==================")
# =============================================================================
# =============================================================================






# =============================================================================
# Make hourly Dst values from the 1-min Dst:
# =============================================================================
from datetime import datetime, timedelta
import statistics

# Starting date:
test0_date = "2022-09-06T00:00:00.000Z"
test_date = pd.to_datetime(test0_date)
c = np.where(date_time < test_date)

# Start calculations for 24 hours:
hourly_Dst_Burton = np.zeros(24)
hourly_Dst_OBrien_McPherron = np.zeros(24)
hourly_Dst_Burton_1au = np.zeros(24)
hourly_Dst_OBrien_McPherron_1au = np.zeros(24)

for i in range(24):
    #print('--- ', i)
    # Find all the values within 1 hour, e.g. between 00:00 and 01:00 etc.
    c = np.where((date_time > test_date+(i)*timedelta(hours=1)) & (date_time < test_date+(i+1)*timedelta(hours=1)))
    arr = np.array(c)
    #print(arr.shape[1])
    newarr = arr.reshape(arr.shape[1], -1)

    #print('new shape: ', newarr.shape)
    #print(len(newarr))
    #print(min(newarr)[0])
    #print(max(newarr)[0])
    
    #print('First value: ', Dst_Burton[min(newarr)])
    #print('Last value: ', Dst_Burton[max(newarr)])
    mean_Dst = statistics.mean(Dst_Burton[min(newarr)[0]:max(newarr)[0]+1])

    #print('first: ', Dst_Burton[min(newarr)[0]])
    #print('last: ', Dst_Burton[max(newarr)[0]])
    #print(mean_Dst)
    hourly_Dst_Burton[i] = mean_Dst
    
    #print('First value: ', Dst_Burton_1au[min(newarr)])
    #print('Last value: ', Dst_Burton_1au[max(newarr)])
    mean_Dst_1au = statistics.mean(Dst_Burton_1au[min(newarr)[0]:max(newarr)[0]+1])
    #print('first: ', Dst_Burton_1au[min(newarr)[0]])
    #print('last: ', Dst_Burton_1au[max(newarr)[0]])
    #print(mean_Dst_1au)
    hourly_Dst_Burton_1au[i] = mean_Dst_1au
    
    #print('First value: ', Dst_OBrien_McPherron[min(newarr)])
    #print('Last value: ', Dst_OBrien_McPherron[max(newarr)])
    mean_Dst2 = statistics.mean(Dst_OBrien_McPherron[min(newarr)[0]:max(newarr)[0]+1])
    #print('first: ', Dst_OBrien_McPherron[min(newarr)[0]])
    #print('last: ', Dst_OBrien_McPherron[max(newarr)[0]])
    #print(mean_Dst2)
    hourly_Dst_OBrien_McPherron[i] = mean_Dst2
    
    #print('First value: ', Dst_OBrien_McPherron_1au[min(newarr)])
    #print('Last value: ', Dst_OBrien_McPherron_1au[max(newarr)])
    mean_Dst2_1au = statistics.mean(Dst_OBrien_McPherron_1au[min(newarr)[0]:max(newarr)[0]+1])
    #print('first: ', Dst_OBrien_McPherron_1au[min(newarr)[0]])
    #print('last: ', Dst_OBrien_McPherron_1au[max(newarr)[0]])
    #print(mean_Dst2_1au)
    hourly_Dst_OBrien_McPherron_1au[i] = mean_Dst2_1au
    

print(" ")
print(" ")
print(" ")
print("=== HOURLY Dst - START ===")

#Check:
#print(hourly_Dst_Burton[9:])

print("========= Burton et al., 1975 model - 1 hour Dst =========")
print("Burton et al., 1975 Dst: ")
print("min Dst: ", "%4.1f" % min(hourly_Dst_Burton[9:]))
print("max Dst: ", "%4.1f" % max(hourly_Dst_Burton[9:]))
print("+++ extrapolate to 1au: +++")
print("min Dst: ", "%4.1f" % min(hourly_Dst_Burton_1au[9:]))
print("max Dst: ", "%4.1f" % max(hourly_Dst_Burton_1au[9:]))
print("==================")
print("===== O'Brien and McPherron, 2000 model - 1 hour Dst =====")
print("O'Brien and McPherron, 2000 Dst:")
print("min Dst: ", "%4.1f" % min(hourly_Dst_OBrien_McPherron[9:]))
print("max Dst: ", "%4.1f" % max(hourly_Dst_OBrien_McPherron[9:]))
print("+++ extrapolate to 1au: +++")
print("min Dst: ", "%4.1f" % min(hourly_Dst_OBrien_McPherron_1au[9:]))
print("max Dst: ", "%4.1f" % max(hourly_Dst_OBrien_McPherron_1au[9:]))
print("==================")

print("=== HOURLY Dst - END ===")



#print(hourly_Dst_Burton)
#print(hourly_Dst_OBrien_McPherron)
#print(hourly_Dst_Burton_1au)
#print(hourly_Dst_OBrien_McPherron_1au)


# =============================================================================
# plt.plot(hourly_Dst_Burton, 'r')
# plt.plot(hourly_Dst_OBrien_McPherron, 'g')
# plt.plot(hourly_Dst_Burton_1au, 'b')
# plt.plot(hourly_Dst_OBrien_McPherron_1au, 'orange')
# 
# =============================================================================
# =============================================================================


# =============================================================================
# Create the hourly datetime centered at XX:30 starting on 2022-09-06 00:30:00
# =============================================================================
start_date_new = datetime.strptime("2022-09-06 00:30:00", "%Y-%m-%d %H:%M:%S")
end_date_new = start_date_new + timedelta(hours=1)
plot_dates = pd.date_range(start_date_new, periods=24, freq='1H').tolist()
# =============================================================================






#---
# Plot 1:


fig, ax = plt.subplots(figsize=(16,8))


plt.title(str(var) + " SOLO - 2 Dst models [Burton et al., 1975] and [OBrien and McPherron, 2000]")
# plt.xlabel('Dst [nT]')
plt.ylabel('Dst [nT]')


# =============================================================================
# plt.plot(date_time, Dst_Burton, '-r', label='Burton et al., 1975')
# plt.plot(date_time, Dst_OBrien_McPherron, '-g', label='OBrien and McPherron, 2000')
# plt.plot(date_time, Dst_Burton_1au, '--r', label='Burton et al., 1975 - 1au')
# plt.plot(date_time, Dst_OBrien_McPherron_1au, '--g', label='OBrien and McPherron, 2000 - 1au')
# =============================================================================


plt.plot(plot_dates, hourly_Dst_Burton, '-ro', label='Burton et al., 1975')
plt.plot(plot_dates, hourly_Dst_OBrien_McPherron, '-go', label='OBrien and McPherron, 2000')
plt.plot(plot_dates, hourly_Dst_Burton_1au, '--r', label='Burton et al., 1975 - 1au')
plt.plot(plot_dates, hourly_Dst_OBrien_McPherron_1au, '--g', label='OBrien and McPherron, 2000 - 1au')


plt.ylim(-120,120)
#plt.savefig('test_plot1.png', dpi=600)
#files.download("test_plot1.png") 
plt.legend(loc='lower left')


#---
ax.grid(visible=True, which='major', linestyle='--', color='gray')
ax.grid(visible=True, which='minor', linestyle='--', color='gainsboro', axis='both')

ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
#---

ax.yaxis.set_major_locator(MultipleLocator(50))
#ax.yaxis.set_minor_locator(MultipleLocator(100))
#---
#plt.tight_layout()
filename = 'Dst_2_models_' + str(var) +'_SOLO_1minute_hourly.png'
plt.savefig(oFilePath + filename, dpi=300)

plt.show()
#---




#---
# Plot 2 - Multi Plot:
fig = plt.figure(2,figsize=(16,12))
plt.subplots_adjust(hspace=0)

#--- (1):
ax = fig.add_subplot(411)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.grid(visible=True, which='major', linestyle='--', color='gray')
ax.grid(visible=True, which='minor', linestyle='--', color='gainsboro', axis='both')
label_size = 18
ax.tick_params(labelsize=label_size)

sdate0 = datetime.strptime("2022-09-06 00:30:00", "%Y-%m-%d %H:%M:%S")
edate0 = datetime.strptime("2022-09-06 23:30:00", "%Y-%m-%d %H:%M:%S")
ax.set_xlim([sdate0-timedelta(hours=1.50), edate0+timedelta(hours=1.50)])
#ax.set_xticklabels([])

plt.plot(date_time, density, '-m', markersize = 4, label='Density')
plt.plot(date_time, density_1au, '--', color = "gray")
plt.ylabel('$N_{p}$ [$cm^{-3}$]', fontsize=18)

#--- (2):
ax = fig.add_subplot(412)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.grid(visible=True, which='major', linestyle='--', color='gray')
ax.grid(visible=True, which='minor', linestyle='--', color='gainsboro', axis='both')
ax.tick_params(labelsize=label_size)
sdate0 = datetime.strptime("2022-09-06 00:30:00", "%Y-%m-%d %H:%M:%S")
edate0 = datetime.strptime("2022-09-06 23:30:00", "%Y-%m-%d %H:%M:%S")
ax.set_xlim([sdate0-timedelta(hours=1.50), edate0+timedelta(hours=1.50)])
plt.plot(date_time, speed, '-k', markersize = 4, label='Speed')
plt.ylabel('$V_{SW}$ [km/s]', fontsize=18)

#--- (3):
ax = fig.add_subplot(413)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y %H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.grid(visible=True, which='major', linestyle='--', color='gray')
ax.grid(visible=True, which='minor', linestyle='--', color='gainsboro', axis='both')
ax.tick_params(labelsize=label_size)
sdate0 = datetime.strptime("2022-09-06 00:30:00", "%Y-%m-%d %H:%M:%S")
edate0 = datetime.strptime("2022-09-06 23:30:00", "%Y-%m-%d %H:%M:%S")
ax.set_xlim([sdate0-timedelta(hours=1.50), edate0+timedelta(hours=1.50)])

#Check:
#range1 = [0:646]
#range2 = [688:990]
#range3 = [1052:1440]

plt.plot(date_time, BzR, '-b', label='B_R')
plt.plot(date_time, BzT, '-r', label='B_T')
plt.plot(date_time, BzN, '-g', label='B_N')

plt.plot(date_time, BzN_1au, '--', color = "gray")

plt.legend(loc='lower left')
plt.ylabel('$B_{i}$ [nT]', fontsize=18)

# =============================================================================
# #--- (4):
# ax = fig.add_subplot(414)
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y \n %H:%M'))
# ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
# ax.grid(visible=True, which='major', linestyle='--', color='gray')
# ax.grid(visible=True, which='minor', linestyle='--', color='gainsboro', axis='both')
# ax.tick_params(labelsize=label_size)
# plt.plot(date_time, Dst_Burton, '--', color = "black", label='Burton et al., 1975')
# plt.plot(date_time, Dst_OBrien_McPherron, '--', color = "gray", label='OBrien and McPherron, 2000')
# plt.plot(date_time, Dst_Burton_1au, '-r', label='Burton et al., 1975 - 1au')
# plt.plot(date_time, Dst_OBrien_McPherron_1au, '-g', label='OBrien and McPherron, 2000 - 1au')
# plt.ylabel('Dst [nT]', fontsize=18)
# =============================================================================

#--- (4):
ax = fig.add_subplot(414)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y \n %H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.grid(visible=True, which='major', linestyle='--', color='gray')
ax.grid(visible=True, which='minor', linestyle='--', color='gainsboro', axis='both')
ax.tick_params(labelsize=label_size)
plt.plot(plot_dates, hourly_Dst_Burton, '--o', color = "black", label='Burton et al., 1975')
plt.plot(plot_dates, hourly_Dst_OBrien_McPherron, '--o', color = "gray", label='OBrien and McPherron, 2000')
plt.plot(plot_dates, hourly_Dst_Burton_1au, '-ro', label='Burton et al., 1975 - 1au')
plt.plot(plot_dates, hourly_Dst_OBrien_McPherron_1au, '-go', label='OBrien and McPherron, 2000 - 1au')
plt.ylabel('Dst [nT]', fontsize=18)


sdate0 = datetime.strptime("2022-09-06 00:30:00", "%Y-%m-%d %H:%M:%S")
edate0 = datetime.strptime("2022-09-06 23:30:00", "%Y-%m-%d %H:%M:%S")
ax.set_xlim([sdate0-timedelta(hours=1.50), edate0+timedelta(hours=1.50)])
ax.set_ylim(-52,+52)
plt.legend(loc='upper left')
filename = 'Output_SOLO_' + str(var) + '_1minute_hourly.png'
plt.savefig(oFilePath + filename, dpi=300)
plt.show()
