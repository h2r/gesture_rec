#!/usr/bin/env python

import rospy
from std_msgs.msg import String


cur_ing = ""
correct = 0
total = 0

def ground_truth_cb(data):
	global cur_ing
	global total
	cur_ing = data.data
	print "ground truth: " + cur_ing
	total += 1

def robot_cb(data):
	global correct
	print 'robot thinks: ' + data.data
	if cur_ing == data.data:
		print True
		correct += 1
	print float(correct)/total



def listener():
	rospy.init_node('listener', anonymous=True)
	rospy.Subscriber('fetch_commands', String, robot_cb)
	rospy.Subscriber('ground_truth', String, ground_truth_cb)
	rospy.spin()

if __name__ == '__main__':
	listener()