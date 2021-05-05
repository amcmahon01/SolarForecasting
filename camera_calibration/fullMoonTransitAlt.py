import ephem
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DayLocator, HourLocator, DateFormatter, drange
moon=ephem.Moon()
obs=ephem.Observer()
# McMurdo
# obs.lon='166:43:42.4'
# obs.lat='-77:50:58.6'
# Barrow
# obs.lon='-156.6156'
# obs.lat='71.3230'
# SGP
#obs.lon='-97.5'
#obs.lat='36.6167'
# BNL
#obs.lon='-72.87'
#obs.lat='40.87'
# Albany
obs.lon='-73.8260'
obs.lat='42.6833'
#
obs.date="2019-12-01"
#obs.date=ephem.next_solstice(obs.date)
moon.compute(obs)
d=np.empty(0)
fmta=np.empty(0)
for i in range(14):
    obs.date = ephem.next_full_moon(obs.date)
    moon.compute(obs)
    obs.date=obs.next_transit(moon)
    moon.compute(obs)
    d=np.append(d,ephem.date(obs.date).datetime())
    fmta=np.append(fmta,moon.alt)
fig,ax=plt.subplots()
ax.plot(d,np.rad2deg(fmta),'o')
ax.set_title('Full Moon Transit at lon='+str(obs.lon)+' lat='+str(obs.lat))
ax.set_ylabel('Altitude [deg]')
plt.show()
