#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import csv

responses = [] #list of all commands

ingredients = ("carduni", \
	"carduni", \
	"pancake_mix", \
	"pancake_mix", \
	"salt", \
	"salt", \
	"garlic", \
	"garlic", \
	"black_pepper", \
	"black_pepper", \
	"olive_oil", \
	"olive_oil")

cur_ing = ""


def csv_parser():
	global responses
	with open('Batch_carduni2.csv', 'rb') as csvfile:
		hit_reader = csv.reader(csvfile)
		x = 0
		for row in hit_reader:
			for i in range(0, len(row)):
				if row[i] == 'Answer.command':
					print i
			#print row[26]
			if row[33] != 'Answer.command':
				#print row[32].split('|')
				responses.append(row[33].split('|'))
	print 'number of responses: ' + str(len(responses))


def talker():
	global cur_ing
	pub = rospy.Publisher('speech_recognition', String, queue_size=1)
	pub2 = rospy.Publisher('ground_truth', String, queue_size=1)
	rospy.init_node('sim_talker', anonymous=False)
	rate=rospy.Rate(0.5)
	minirest = rospy.Rate(10)
	for i in range(0, len(responses)):
		print 'recipe #%d' % i
		for j in range(0, len(responses[i])):
			if not rospy.is_shutdown():
				rate.sleep()
				if j%2 == 0:
					print ingredients[j] + '; ' + responses[i][j]
					pub2.publish(ingredients[j])
					pub.publish(responses[i][j].replace('.','').replace(',','').replace('?','').lower())
				minirest.sleep()
				#pub.publish(responses[i][j].replace('.','').replace(',','').replace('?','').lower())


if __name__ == '__main__':
	try:
		csv_parser()
		talker()
	except rospy.ROSInterruptException:
		exit()