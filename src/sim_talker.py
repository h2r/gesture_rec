#!/usr/bin/env python

import rospy
from std_msgs.msg import String


commands = ("can you pass me the chocolate", \
	"can you hand me the butter", \
	"give me the eggs please", \
	"hand me the sugar", \
	"I want the vanilla", \
	"please pass me the flour", \
	"I need the salt")

def talker():
	pub = rospy.Publisher('speech_recognition', String, queue_size=1)
	rospy.init_node('sim_talker', anonymous=False)
	rate=rospy.Rate(0.1)
	count = 0
	while not rospy.is_shutdown():
		rate.sleep()
		pub.publish(commands[count])
		count += 1

if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass