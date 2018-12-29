#!/usr/bin/python
# -*- coding: latin-1 -*-
import matplotlib.pyplot as plt
import numpy as np
import re

clock = 0
time = []
stationerror = []
lateralerror = []
headingerror = []

velocity_station_error = []
velocity_lateral_error = []
truevelocity_station = []
truevelocity_lateral = []

acceleration_station_error = []
acceleration_lateral_error = []
trueacceleration_station = []
trueacceleration_lateral = []

gps_heading = []
estimate_heading = []

perception_error = []

input = open('/apollo/data/log/localization.INFO', 'r')
data = input.readlines()

begintime = 0.0
for line in data:
    if re.findall('Timestamp', line):
        line = line.split()
        if len(time):
            detatime = (float)(line[-1].split('[')[1].split(']')[0]) \
                                   - begintime
            time.append(detatime)
        else:
            begintime = (float)(line[-1].split('[')[1].split(']')[0])
            time.append(0.0)

for line in data:
    if re.findall('position error', line):
        try:
            line = line.split(',')
            station = line[5].split('[')[1].split(']')[0]
            lateral = line[6].split('[')[1].split(']')[0]
            stationerror.append((float)(station))
            lateralerror.append((float)(lateral))
        except:
            pass


for line in data:
    if re.findall('heading error', line):
        try:
            line = line.split()
            heading = line[-1].split('[')[1].split(']')[0]
            headingerror.append(57.3*(float)(heading))
        except:
            pass

true_heading = []
calculate_heading = []
for line in data:
    if re.findall('heading error', line):
        try:
            line = line.split(',')
            truehead = line[0].split()[-1].split('[')[1].split(']')[0]
            headerror = line[1].split('[')[1].split(']')[0]
            true_heading.append(57.3*(float)(truehead))
            calculate_heading.append(57.3*((float)(truehead)+(float)(headerror)))
        except:
            pass

for line in data:
    if re.findall('True velocity', line):
        try:
            line = line.split(',')
            true_x = line[1].split('[')[1].split(']')[0]
            true_y = line[2].split('[')[1].split(']')[0]
            error_x = line[4].split('[')[1].split(']')[0]
            error_y = line[5].split('[')[1].split(']')[0]
            velocity_station_error.append((float)(error_x))
            velocity_lateral_error.append((float)(error_y))
            truevelocity_station.append((float)(true_x))
            truevelocity_lateral.append((float)(true_y))
        except:
            pass


for line in data:
    if re.findall('True acceleration', line):
        try:
            a = line.split('[')[2].split(']')[0]
            line = line.split(',')
            true_x = line[1].split('[')[1].split(']')[0]
            true_y = line[2].split('[')[1].split(']')[0]
            error_x = line[4].split('[')[1].split(']')[0]
            error_y = line[5].split('[')[1].split(']')[0]
            acceleration_station_error.append((float)(error_x))
            acceleration_lateral_error.append((float)(error_y))
            trueacceleration_station.append((float)(true_x))
            trueacceleration_lateral.append((float)(true_y))
        except:
            pass

for line in data:
    if re.findall('On registrating', line):
        try:
            line = line.split(',')
            error = line[-1].split('[')[1].split(']')[0]
            perception_error.append((float)(error))
        except:
            pass

plt.figure('perception Error')
plt.plot(perception_error, color='blue', label='$station$')
plt.legend()
#####################################################
minVal = min(len(tuple(time)),len(tuple(stationerror)),len(tuple(lateralerror)),\
                len(tuple(headingerror)),len(tuple(velocity_station_error)),\
                len(tuple(velocity_lateral_error)),len(tuple(truevelocity_station)))

minVal = min(minVal,len(tuple(truevelocity_lateral)),len(tuple(acceleration_station_error)),\
                 len(tuple(acceleration_lateral_error)),len(tuple(trueacceleration_station)),\
                 len(tuple(trueacceleration_lateral)),len(tuple(true_heading)),len(tuple(calculate_heading)))
                 
while len(tuple(time)) > minVal :
    time.pop()

while len(tuple(stationerror)) > minVal :
    stationerror.pop()

while len(tuple(lateralerror)) > minVal :
    lateralerror.pop()

while len(tuple(headingerror)) > minVal :
    headingerror.pop()

while len(tuple(velocity_station_error))> minVal :
    velocity_station_error.pop()

while len(tuple(velocity_lateral_error))> minVal :
    velocity_lateral_error.pop()

while len(tuple(truevelocity_station)) > minVal :
    truevelocity_station.pop()

while len(tuple(truevelocity_lateral)) > minVal :
    truevelocity_lateral.pop()

while len(tuple(acceleration_station_error)) > minVal :
    acceleration_station_error.pop()

while len(tuple(acceleration_lateral_error)) > minVal :
    acceleration_lateral_error.pop()

while len(tuple(trueacceleration_station)) > minVal :
    trueacceleration_station.pop()

while len(tuple(trueacceleration_lateral)) > minVal :
    trueacceleration_lateral.pop()

while len(tuple(true_heading)) > minVal :
    true_heading.pop()

while len(tuple(calculate_heading)) > minVal :
    calculate_heading.pop()
#####################################################


plt.figure('Position Error')
plt.title('Position Error Analysis')
plt.plot(time, stationerror, color='blue', label='$station$')
plt.plot(time, lateralerror, color='red', label='$lateral$')
plt.plot(time, headingerror, color='black', label='$heading$')
plt.legend()

# 设置坐标轴名称
plt.xlabel('time(s)')
plt.ylabel('error(m)')
# plt.show()

plt.figure('Velocity Error')
plt.title('Velocity Error Analysis')
plt.plot(time, velocity_station_error, color='blue', label='$StationError$')
plt.plot(time, velocity_lateral_error, color='red', label='$LateralError$')
# plt.plot(time,truevelocity_station,color='black',label='$TrueStation$')
# plt.plot(time,truevelocity_lateral,color='green',label='$TrueLateral$')
plt.legend()
# 设置坐标轴名称
plt.xlabel('time(s)')
plt.ylabel('error(m/s)')
# plt.show()

plt.figure('Acceleration Error')
plt.title('Acceleration Error Analysis')
plt.plot(time, acceleration_station_error,
         color='blue', label='$StationError$')

plt.legend()
# 设置坐标轴名称
plt.xlabel('time(s)')
plt.ylabel('error(m/s^2)')
#plt.show()

plt.figure('heading')
plt.title('trueheading and calculateheading')
plt.plot(time, true_heading,
         color='blue', linewidth=2.0,label='$true_heading$')
plt.plot(time, calculate_heading,
         color='red', linewidth=2.0,label='$calculate_heading$')

plt.legend()
# 设置坐标轴名称
plt.xlabel('time(s)')
plt.ylabel('c')
plt.show()

# 画gps图
trueX = []
trueY = []
for line2 in data:
    if re.findall('True position', line2):
        try:
            line2 = line2.split(',')
            x = line2[1].split('[')[1].split(']')[0]
            y = line2[2].split('[')[1].split(']')[0]
            trueX.append((float)(x))
            trueY.append((float)(y))
        except:
            pass



###############################################
#minVal = min(minVal,len(trueX),len(trueY))
while len(time) > minVal :
    time.pop()

while len(tuple(trueX)) > minVal :
    trueX.pop()

while len(tuple(trueY)) > minVal :
    trueY.pop()
###############################################

# print(trueX)
plt.figure('GPSX')
plt.plot(time, trueX, '.r')
plt.plot(time, trueX, 'b')

plt.figure('GPSY')
plt.plot(time, trueY, '.b',)
plt.plot(time, trueY, 'r')

#plt.show()

