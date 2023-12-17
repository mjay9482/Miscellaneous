#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:25:41 2023

@author: mrityunjay
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.children = []
        self.parent = None

class RRTAlgorithm:
    def __init__(self, start, goal, num_iterations, grid, step_size):
        self.random_tree = Node(start[0], start[1])
        self.goal_node = Node(goal[0], goal[1])
        self.nearest_node = None
        self.iterations = min(num_iterations, 300)
        self.grid = grid
        self.rho = step_size
        self.num_waypoints = 0
        self.nearest_dist = 10000
        self.path_distance = 0
        self.waypoints = []

    def unit_vector(self, start_node, end_node):
        v = np.array([end_node.x - start_node.x, end_node.y - start_node.y])
        u_hat = v / np.linalg.norm(v)
        return u_hat

    def add_child(self, x, y):
        if x == self.goal_node.x:
            self.nearest_node.children.append(self.goal_node)
            self.goal_node.parent = self.nearest_node
        else:
            temp_node = Node(x, y)
            self.nearest_node.children.append(temp_node)
            temp_node.parent = self.nearest_node

    def sample_point(self):
        x = random.randint(1, grid.shape[1] - 1)
        y = random.randint(1, grid.shape[0] - 1)
        point = np.array([x, y])
        return point

    def steer_to_point(self, start_node, end_point):
        offset = self.rho * self.unit_vector(start_node, end_point)
        point = np.array([start_node.x + offset[0], start_node.y + offset[1]])
        if point[0] >= grid.shape[1]:
            point[0] = grid.shape[1] - 1
        if point[1] >= grid.shape[0]:
            point[1] = grid.shape[0] - 1
        return point

    def is_in_obstacle(self, start_node, end_point):
        u_hat = self.unit_vector(start_node, Node(end_point[0], end_point[1]))
        test_point = np.array([0.0, 0.0])
        for i in range(int(self.rho)):
            test_point[0] = start_node.x + i * u_hat[0]
            test_point[1] = start_node.y + i * u_hat[1]
            if (test_point[0] < 0 or test_point[0] >= self.grid.shape[1] or test_point[1] < 0 or test_point[1] >= self.grid.shape[0]):
                return True
            if self.grid[int(round(test_point[1])), int(round(test_point[0]))] == 1:
                return True
        return False

    def find_nearest(self, root, point):
        if not root:
            return
        dist = self.distance(root, point)
        if dist <= self.nearest_dist:
            self.nearest_node = root
            self.nearest_dist = dist
        for child in root.children:
            self.find_nearest(child, point)

    def distance(self, node, point):
        dist = np.sqrt((node.x - point[0]) ** 2 + (node.y - point[1]) ** 2)
        return dist

    def goal_found(self, point):
        if self.distance(self.goal_node, point) <= self.rho:
            return True

    def reset_nearest_values(self):
        self.nearest_node = None
        self.nearest_dist = 10000

    def retrace_rrt_path(self, goal):
        if goal.x == self.random_tree.x:
            return
        self.num_waypoints += 1
        current_point = np.array([goal.x, goal.y])
        self.waypoints.insert(0, current_point)
        self.path_distance += self.rho
        self.retrace_rrt_path(goal.parent)

# Define grid and start/goal points
grid = np.load('cspace.npy')
start = np.array([100.0, 100.0])
goal = np.array([700.0, 250.0])
num_iterations = 300
step_size = 60

# Create a visualization of the grid with start and goal points
start_location = plt.Circle((start[0], start[1]), step_size/2, color='c', fill=False)
goal_region = plt.Circle((goal[0], goal[1]), step_size, color='b', fill=False)
fig = plt.figure('RRT Algorithm')
plt.imshow(grid, cmap='binary')
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'bo')
ax = fig.gca()
ax.add_patch(start_location)
ax.add_patch(goal_region)
plt.xlabel('X-axis (m)')
plt.ylabel('Y-axis (m)')
plt.show()

# Initialize the RRT algorithm
rrt = RRTAlgorithm(start, goal, num_iterations, grid, step_size)

for i in range(rrt.iterations):
    rrt.reset_nearest_values()
    print('Iteration:', i)
    point = rrt.sample_point()
    rrt.find_nearest(rrt.random_tree, point)
    new_point = rrt.steer_to_point(rrt.nearest_node, point)
    is_in_obstacle = rrt.is_in_obstacle(rrt.nearest_node, new_point)
    if not is_in_obstacle:
        rrt.add_child(new_point[0], new_point[1])
        plt.pause(0.1)
        plt.plot([rrt.nearest_node.x, new_point[0]], [rrt.nearest_node.y, new_point[1]], 'go', linestyle="--")
        if rrt.goal_found(new_point):
            rrt.add_child(goal[0], goal[1])
            print('Goal found!')
            break

rrt.retrace_rrt_path(rrt.goal_node)
rrt.waypoints.insert(0, start)
print("Number of waypoints:", rrt.num_waypoints)
print("Path Distance:", rrt.path_distance)
print("Waypoints:", rrt.waypoints)

for i in range(len(rrt.waypoints) - 1):
    plt.plot([rrt.waypoints[i][0], rrt.waypoints[i + 1][0]], [rrt.waypoints[i][1], rrt.waypoints[i + 1][1]], 'ro', linestyle='--')
    plt.pause(0.1)

