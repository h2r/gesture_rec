#!/usr/bin/env python

import rospy
from std_msgs.msg import String


last_truth = ""
cur_truth = ""
cur_predict = ""
correct = 0
total = 0

def ground_truth_cb(data):
	global cur_truth
	global total
	global correct
	global last_truth
	last_truth = cur_truth
	cur_truth = data.data
	print "ground truth: " + cur_truth
	#print 'robot thinks: ' + cur_predict
	
	total += 1

	#print float(correct)/total


def robot_cb(data):
	global cur_predict
	global correct
	print 'robot thinks (cb): ' + data.data
	cur_predict = data.data
	if cur_truth == cur_predict:
		correct += 1
		print True

	print float(correct)/total





def listener():
	rospy.init_node('listener', anonymous=True)
	rospy.Subscriber('fetch_commands', String, robot_cb, queue_size=2)
	rospy.Subscriber('ground_truth', String, ground_truth_cb)
	rospy.spin()

if __name__ == '__main__':
	listener()