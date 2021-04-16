import numpy as np
import csv




def flip(bias):
	return 1 if np.random.random()<bias else 0

def generateFile(choice,n):
	fields = ["c1","c2","c3","c4","c5"]
	with open('bias.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(fields)
		for j in range(n):
			p=[]
			for i in choice:
				p.append(flip(i))
			writer.writerow(p)

def findBias():
	row=[]
	with open('bias.csv', 'r') as csvfile:
		reader=csv.reader(csvfile)
		fields=next(reader)
		sum = np.zeros(len(fields))
		rows=0
		for row in reader:
			rows+=1
			sum+=np.array([int(i) for i in row] )
		return (sum/rows)

bias_coins = np.random.random(10)
choice = np.random.choice(bias_coins,5)
actual_bias=np.array(choice)
generateFile(choice,100)
calculated_bias = findBias()
print(np.mean(abs(calculated_bias-actual_bias)))
