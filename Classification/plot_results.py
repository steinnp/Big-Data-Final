#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
##
## DM2583 Big Data in Media Technology
## Final project
##
## Carlo Rapisarda
## carlora@kth.se
##
## Plot Results
## Oct. 10, 2017
##


import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.ticker import FuncFormatter, MaxNLocator


def plot_raw_classification(csv_file_path):
	values = []
	with open(csv_file_path, "r") as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		for line in reader:
			values.append(float(line[1]))
	plot_histogram(values, 15, 'Evaluation', 'Amount')



def plot_histogram(values, bars, xlabel, ylabel):

	fig, ax = plt.subplots()
	n, bins = np.histogram(values, bars)

	# get the corners of the rectangles for the histogram
	left = np.array(bins[:-1])
	right = np.array(bins[1:])
	bottom = np.zeros(len(left))
	top = bottom + n

	# we need a (numrects x numsides x 2) numpy array for the path helper
	# function to build a compound path
	XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

	# get the Path object
	barpath = path.Path.make_compound_path_from_polys(XY)

	# make a patch out of it
	patch = patches.PathPatch(barpath)
	ax.add_patch(patch)

	# update the view limits
	ax.set_xlim(left[0], right[-1])
	ax.set_ylim(bottom.min(), top.max())

	labels = ['Negative', 'Neutral', 'Positive']

	def format_fn(tick_val, tick_pos):
		if int(tick_val) in range(0,3):
			return labels[int(tick_val)]
		else:
			return ''

	ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()


# plot_raw_classification("./480ktrump_results.csv")
