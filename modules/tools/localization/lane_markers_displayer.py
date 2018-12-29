#!/usr/bin/python

from math import cos
from math import sin
from modules.localization.proto import odometry_lane_marker_pb2
import sys
import matplotlib.pyplot as plt
import numpy as np
import re


def read_lane_markers_data(filename):
    left_lane_markers_points_group = []
    right_lane_markers_points_group = []
    input = open(filename, 'rb')
    data = input.read()
    target = odometry_lane_marker_pb2.OdometryLaneMarkersPack()
    target.ParseFromString(data)
    left_lane_marker_group = target.lane_markers[0]
    right_lane_marker_group = target.lane_markers[1]
    for lane_marker in left_lane_marker_group.lane_marker:
        for point in lane_marker.points:
            left_lane_markers_points_group.append(
                [point.position.x, point.position.y])

    for lane_marker in right_lane_marker_group.lane_marker:
        for point in lane_marker.points:
            right_lane_markers_points_group.append(
                [point.position.x, point.position.y])

    input = open('/apollo/data/log/localization.INFO', 'r')
    data = input.readlines()
    Estimateposition = []

    for line in data:
        if re.findall('Estimate position', line):
            try:
                line = line.split(',')
                positionX = line[1].split('[')[1].split(']')[0]
                positionY = line[2].split('[')[1].split(']')[0]
                Estimateposition.append([positionX, positionY])
            except:
                pass

    GPS = []
    for line2 in data:
        if re.findall('True position', line2):
            try:
                line2 = line2.split(',')
                x = line2[1].split('[')[1].split(']')[0]
                y = line2[2].split('[')[1].split(']')[0]
                GPS.append([(float)(x), (float)(y)])
            except:
                pass

    plot_lane(left_lane_markers_points_group, right_lane_markers_points_group,GPS,Estimateposition)


def plot_lane(left_points, right_points,position,GPS):
    """plot the points list in matplotlib 
    Args:
        left_points: points list of left lane marker group
        right_points: points list of right lane marker group
    """
    x_left = np.array(left_points)[:, 0]
    y_left = np.array(left_points)[:, 1]
    plt.scatter(x_left, y_left,color='b')
    x_right = np.array(right_points)[:, 0]
    y_right = np.array(right_points)[:, 1]
    plt.scatter(x_right, y_right,color='b')
    position_x = np.array(position)[:, 0]
    position_y = np.array(position)[:, 1]
    plt.scatter(position_x, position_y,color='g')
    gps_x = np.array(GPS)[:, 0]
    gps_y = np.array(GPS)[:, 1]
    plt.scatter(gps_x, gps_y,color='r')
    plt.show()


def main():
    """main entry of lane_markers_displayer
    Args:
        argv[1]:raw OdometryLanemarker bin file input to display 
    """
    if(len(sys.argv) != 2):
        print(
            "Please provide --argv[1]:raw OdometryLanemarker bin file to display")
        sys.exit()
    read_lane_markers_data(sys.argv[1])


if __name__ == '__main__':
    main()
