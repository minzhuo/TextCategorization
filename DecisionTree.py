from numpy import *
from object_json import *
from copy import *
from math import log
import pdb
import operator
import treePlotter
import re, sys
import numpy as np
import pylab as pl
from sklearn import tree

max_depth = 1000

def Draw(x1, y1, x2, y2):
	x1 = [1, 2, 3, 4, 5]# Make x, y arrays for each graph
	y1 = [1, 4, 9, 16, 25]
	x2 = [1, 2, 4, 6, 8]
	y2 = [2, 4, 8, 12, 16]
	 
	plot1 = pl.plot(x1, y1, 'r', label = 'Train Set')# use pylab to plot x and y
	plot2 = pl.plot(x2, y2, 'g', label = 'Test set')
	 
	pl.title('Learning curves')# give plot a title
	pl.xlabel('Tree size')# make axis labels
	pl.ylabel('% correct')
	 
	 
	pl.xlim(0.0, 9.0)# set axis limits
	pl.ylim(0.0, 30.)

	pl.legend('best')
	pl.show()

def SklearnDecsionTree(X,Y):
	iris = load_iris()
	clf = tree.DecisionTreeClassifier()
	
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

	return 1, sample, train_labels, test_sample, test_labels, labels

def Majority(List):
	label = {}
	maxLable = -1
	for l in List:
		if l not in label:
			label[l] = 0
		label[l] += 1
		if maxLable == -1:
			maxLable = l
		if label[maxLable] < label[l]:
		 	maxLable = l
	return maxLable

def Entropy(sample):
	numSample = len(sample)
	classificaton  = {}
	for s in sample:
		a = s[-1]

		if s[-1] in classificaton.keys():
			classificaton[s[-1]] += 1
		else :
			classificaton[s[-1]] = 1
	entropy = 0.0
	for key in classificaton:
		p = classificaton[key] / float(numSample)
		entropy -= p * log (p ,2)

	return entropy

def BestFeature(sample):
	numFeatures = len(sample[0]) - 1;
	bestfeature = -1
	lowestEntropy = 2
	for index in range(numFeatures):
		featureList = [s[index] for s in sample]
		allValue = set (featureList)
		entropy = 0.0
		for value in allValue:
			subSample = SplitSamlpe(sample, index, value)
			p = len(subSample) / float(len(sample))
			entropy += p * Entropy(subSample)
		if entropy < lowestEntropy:
			lowestEntropy = entropy
			bestfeature = index
	print bestfeature
	return bestfeature

def SplitSamlpe(sample, index, value):
	subSample = []
	for s in sample:
		if s[index] != value:
			continue
		tempList = s[:index]
		tempList.extend(s[index + 1:])
		subSample.append(tempList)
	return subSample

def DTL(sample, label, depth):
	List = [s[-1] for s in sample] 
	if depth >= max_depth or len(sample[0]) == 1:
		return Majority(List)
	if List.count(List[0]) == len(List): 
		return List[0]    	

	print "choosing  best feature"
	bestfeature = BestFeature(sample)
	print "    best feature:", label[bestfeature]
	Tree  = {label[bestfeature]:{}}
	featureList = [s[bestfeature] for s in sample]
	allValue = set (featureList)
	for value in allValue:
		Tree[label[bestfeature]][value] = DTL(SplitSamlpe(sample, bestfeature, value), label, depth + 1)

	return Tree

def preditor(sample, Tree, labels, index):
	if isinstance(Tree,int):
		return Tree
	key = Tree.keys()[0] 
	value =  sample[index][labels.index(key)]
	return preditor(sample, Tree[key][value], labels, index)

def Test(sample, Tree, labels, sample_labels):
	numSample = len(sample)
	subTree = Tree
	error_num = 0
	for i in range(numSample):
		pre = preditor(sample, Tree, labels, i)
		print type(pre)
		if sample_labels[i] == pre:
			error_num += 1

	error_rate = float(error_num) / numSample
	print "Error rate:", error_rate

def main():
	depth, sample, train_labels, test_sample, test_labels, labels = readFile()
	TotalTree = []
	for depth in range(2, 40):
		Tree = DTL(sample, labels, depth)
		TotalTree.append(Tree)
	print "For Train Sample:"
	Test(sample, Tree, labels, train_labels)
	print "For Train Sample:"
	Test(test_sample, Tree, labels, test_labels)
	
if __name__ == "__main__":
    main()

