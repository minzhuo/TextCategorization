from numpy import *
from object_json import *
from copy import *
from math import log
import pdb
import operator
import treePlotter
import re, sys

def SKlearnNaiveBeyas(parameters,labels):
	a = 1
	a = 3

def readFile():
	labels = []
	train_labels = []
	test_labels = []
	file_label = open('word.txt', 'r')
	file_data = open('trainData.txt', 'r')
	file_datalabel = open('trainLabel.txt', 'r')
	file_testData = open('testData.txt', 'r')
	file_testLabel = open('testLabel.txt','r')

	for line in file_label.readlines():
		m = re.match(r'^([a-zA-Z]+)$', line)
		labels.append(m.group(0))
	

	for line in file_datalabel.readlines():
		m = re.match(r'^([0-9]+)$', line)
		train_labels.append(int(m.group(0)))

	for line in file_testLabel.readlines():
		test_labels.append(int(line.strip()))

	sample = [[0] * (len(labels) + 1) for n in range(len(train_labels))]
	test_sample = [[0] * (len(labels) + 1) for n in range(len(test_labels))]

	for i in range(len(train_labels)):
		sample[i][len(labels)] =  train_labels[i]

	for i in range(len(test_labels)):
		test_sample[i][len(labels)] = test_labels[i]

	for line in file_data.readlines(): 
		m = re.match(r'^([0-9]+)\t([0-9]+)$', line)
		current_index = int(m.group(1))
		current_attribute = int(m.group(2))
		sample[current_index - 1][current_attribute - 1] = 1

	for line in file_testData.readlines(): 
		number = line.split('\t')
		current_index = int(number[0])
		current_attribute = int(number[1])
		test_sample[current_index - 1][current_attribute - 1] = 1

	return  sample, train_labels, test_sample, test_labels, labels

def Parameters(sample, train_labels, labels):
	classification = [1, 2]
	parameters = [[0] * len(labels) for n in range(2)]
	print len(labels)
	for i ,value in enumerate(classification):
		submatrix = []
		for s in sample:
			if s[-1] == value:
				submatrix.append(s)
		for index in range(len(labels)):
			length = len(submatrix)
			number = [sub[index] for sub in submatrix].count(1)
			parameters[i][index] = float(number + 1) / (length + 2)

	return parameters

def predictor(parameters, sample):
	prob_1 = 0.0
	prob_2 = 0.0
	for index in range(len(parameters[0])):
		if sample[index] == 1:
			prob_1 += log(parameters[0][index], 2)
			prob_2 += log(parameters[1][index], 2)
		else :
			prob_1 += log(1 - parameters[0][index], 2)
			prob_2 += log(1 - parameters[1][index], 2)
	if prob_1 < prob_2:
		return 2
	return 1

def Test(parameters, sample, labels):
	error_rate = 0.0
	error_number = 0
	for index in range(len(sample)):
		if labels[index] != predictor(parameters,sample[index]):
			error_number += 1
	error_rate = float(error_number) / len(labels)
	print "Correct Rate:",  (1 - error_rate) * 100, "%"

def discriminativeWord(parameters,labels):
	word = []
	for index in range(len(parameters[0])):
		word.append((abs(log(parameters[1][index], 2) - log(parameters[0][index], 2)), labels[index]))
	word.sort(reverse = True)
	print "---------------Top 10 Discriminative Word ------------------"
	for i in range(10):
		print "           ",i + 1, "   :", word[i][-1], "   ", word[i][0]

def main():
	sample, train_labels, test_sample, test_labels, labels = readFile()
	parameters = Parameters(sample, train_labels, labels)
	print "---------------Navie Beyas Model------------------"
	print "For train sample:"
	Test(parameters, sample, train_labels)
	print "For Test sample:"
	Test(parameters, test_sample, test_labels)
	discriminativeWord(parameters, labels)

if __name__ == "__main__":
    main()
		