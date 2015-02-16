#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import csv
import random

responses = [] #list of all commands

#ingredients = ("artichokes", \
#	"artichokes", \
#	"parsley", \
#	"parsley", \
#	"bread_crumbs", \
#	"bread_crumbs", \
#	"parmesan_cheese", \
#	"parmesan_cheese", \
#	"olive_oil", \
#	"olive_oil")

#ingredients = ("carduni", \
#	"carduni", \
#	"pancake_mix", \
#	"pancake_mix", \
#	"salt", \
#	"salt", \
#	"garlic", \
#	"garlic", \
#	"black_pepper", \
#	"black_pepper", \
#	"olive_oil", \
#	"olive_oil")

#ingredients = ('chocolate', \
#	'butter', \
#	'eggs', \
#	'white_sugar', \
#	'vanilla', \
#	'all-purpose_flour', \
#	'salt')

ingredients = ('puff_pastry', \
	'puff_pastry', \
	'spinach', \
	'spinach', \
	'onion', \
	'onion', \
	'garlic', \
	'garlic', \
	'olive_oil', \
	'olive_oil', \
	'ricotta_cheese', \
	'ricotta_cheese', \
	'feta_cheese', \
	'feta_cheese', \
	'salt', \
	'salt', \
	'eggs', \
	'eggs', \
	'sesame_seeds', \
	'sesame_seeds' )

'''
ingredients = ('noodles', \
	'noodles', \
	'garlic', \
	'garlic', \
	'ginger', \
	'ginger', \
	'soy_sauce', \
	'soy_sauce', \
	'brown_sugar', \
	'brown_sugar', \
	'black_pepper', \
	'black_pepper', \
	'olive_oil', \
	'olive_oil', \
	'sesame_oil', \
	'sesame_oil', \
	'onion', \
	'onion', \
	'celery', \
	'celery', \
	'cabbage', \
	'cabbage' )
'''
ingredients = ('chickpeas', \
	'chickepeas', \
	'parsley', \
	'parsley', \
	'cilantro', \
	'cilantro', \
	'garlic', \
	'garlic', \
	'coriander', \
	'coriander', \
	'cumin', \
	'cumin', \
	'black_pepper', \
	'black_pepper', \
	'paprika', \
	'paprika', \
	'salt', \
	'salt', \
	'lemon', \
	'lemon', \
	'olive_oil', \
	'olive_oil', \
	'sesame_seeds', \
	'sesame_seeds' )

cur_ing = ""


def csv_parser():
	global responses
	with open('Batch_falafel.csv', 'rb') as csvfile:
		hit_reader = csv.reader(csvfile)
		x = 0
		for row in hit_reader:
			for i in range(0, len(row)):
				if row[i] == 'Answer.command':
					print i
			if row[28] != 'Answer.command':
				#print row[32].split('|')
				responses.append(row[28].split('|'))
	print 'number of responses: ' + str(len(responses))


def talker():
	global cur_ing
	pub = rospy.Publisher('speech_recognition', String, queue_size=1)
	pub2 = rospy.Publisher('ground_truth', String, queue_size=1)
	rospy.init_node('sim_talker', anonymous=False)
	rate=rospy.Rate(0.5)
	minirest=rospy.Rate(2)
	#f = open('train_data2.txt', 'w')
	for i in range(0, len(responses)):
		print 'recipe #%d' % i
		if i%2 == 1 or True:
			for j in range(0, len(responses[i])):
				if not rospy.is_shutdown():
					if j%2 == 1:
						#print ingredients[j] + ' ' + responses[i][j]
						clean_request = responses[i][j].replace('.','').replace(',','').replace('?','').lower()
						request_list = clean_request.split()
						#for k in range (0, 3):
							#if request_list and request_list: 
								#del request_list[random.randint(0, len(request_list) - 1)]
						clean_request = ' '.join(request_list)
						print ingredients[j] + ' ' + clean_request
						#f.write(ingredients[j] + ' ' + responses[i][j].replace('.','').replace(',','').replace('?','').replace('"','').lower() + '\n')
						
						pub.publish(clean_request)
						#minirest.sleep()
						pub2.publish(ingredients[j])
						rate.sleep()
					#minirest.sleep()
					#pub.publish(responses[i][j].replace('.','').replace(',','').replace('?','').lower())
	#f.close()

if __name__ == '__main__':
	try:
		csv_parser()
		talker()
	except rospy.ROSInterruptException:
		exit()