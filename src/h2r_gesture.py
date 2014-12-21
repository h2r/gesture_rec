#!/usr/bin/env python

import time
import sys
import roslib
roslib.load_manifest("gesture_rec")
import rospy

import tf
import math
import scipy.stats
from numpy import dot
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from std_msgs.msg import Header, String, Float32MultiArray
from tf.transformations import quaternion_inverse, quaternion_matrix

from object_recognition_msgs.msg import RecognizedObjectArray

global left_arm_origin
global right_arm_origin
global head_origin
global left_arm_point
global right_arm_point
global head_point
global speech
global left_foot
global right_foot
global ground_truth
global storage
global write_speech
global tfl
global last_obj
global num_objs
num_objs = -1
last_obj = "unknown"
global scnd_to_lat_obj
scnd_to_lat_obj = "unknown"
storage = None
ground_truth = "None"
speech = []
write_speech = []
global state_dist
state_dist = dict()
global objects
objects = []
containers = {"pink_bowl": "salt", "green_bowl": "pepper", "light_green_bowl": "vanilla", "yellow_bowl": "chocolate"}
#TEMP HACK
#objects = [("pink_box", (1.4,-0.2,-0.5)), ("purple_cylinder", (1.4, 0.05, -0.5))]
#objects = [("light_green_bowl",(1.2, -0.37, -0.37)), ("green_bowl",(1.2, 0.07,-0.37))]
#objects = [("salt", (??, ??, ??)), ]
global t
t = 0.005
global variance
variance = 0.4
global word_probabilities
global vocabulary
global eps
eps = 0.0001
#global user
#user = 1

global pub
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

#David's Dependencies
import operator
import collections
import random

#David's global variables
ingredient_file = 'src/no_repeat_numbered.txt'
recipe_list = []
unigram_init = False
uni_counts = collections.Counter()
past_bigrams = {}
past_trigrams = {}
smoothing_coefficient = 0

#Recipe File Reader
def file_reader():
    global smoothing_coefficient
    vocabulary = collections.Counter()
    f = open(ingredient_file, 'r')
    for line in f:
        clean_line = line.rstrip('\n').encode('utf-8')
        recipe_list.append(clean_line)
        vocabulary[clean_line] += 1
    f.close()
    #Witten-Bell Smoothing coefficient
    smoothing_coefficient = float(len(recipe_list))/(len(recipe_list) + len(vocabulary))

#David's utility functions
def normalize(x): #for dictionary vectors
    total = sum(x.values(), 0.0)
    for key in x:
        x[key] /= total
    return x
def weight(x, weight):
    for key in x:
        x[key] *= weight
    return x

#David's transition functions
def unigram_counter(mod):
    global unigram_init
    global uni_counts
    if unigram_init == False:
        for line in range(0, len(recipe_list)):
            next_ing = recipe_list[line].split(' # ')
            if int(next_ing[0]) != mod:
                uni_counts[next_ing[1].split(',')[0]] += 1.0
        uni_counts = normalize(uni_counts)
        unigram_init = True

    return uni_counts

def bigram_counter(previous_ingredient, mod):

    if previous_ingredient in past_bigrams:
        return past_bigrams[previous_ingredient]
    else:
        b_lam = smoothing_coefficient #need to test for best number
        ni = collections.Counter()
        for line in range(0, len(recipe_list)):
            if previous_ingredient in recipe_list[line]:
                next_ing = recipe_list[line + 1].split(' # ')
                if int(next_ing[0]) != mod:
                    ni[next_ing[1].split(',')[0]] += 1.0
        ni = normalize(ni)

        if not list(ni.items()) or False:
            past_bigrams[previous_ingredient] = weight(unigram_counter(mod), 1.0 - b_lam)
        else:
            past_bigrams[previous_ingredient] = weight(ni, b_lam) + weight(unigram_counter(mod), 1.0 - b_lam)
    
        return past_bigrams[previous_ingredient]

def trigram_counter(prev_ing, prev_ing2, mod):
    input_ings = prev_ing + ":" + prev_ing2
    if input_ings in past_trigrams:
        return past_trigrams[input_ings]
    else:
        t_lam = smoothing_coefficient
        ni = collections.Counter()

        for line in range(0, len(recipe_list)):
            if prev_ing in recipe_list[line] and prev_ing2 in recipe_list[line+1]:
                next_ing = recipe_list[line+2].split(' # ')
                if int(next_ing[0]) != mod:
                    ni[next_ing[1].split(',')[0]] += 1.0

        ni = normalize(ni)
        if not list(ni.items()):
            #return weight(bigram_counter(prev_ing2, mod), 1.0 - t_lam)
            past_trigrams[input_ings] = weight(bigram_counter(prev_ing2, mod), 1.0 - t_lam)
        else:
            #return weight(ni, t_lam) + weight(bigram_counter(prev_ing2, mod), 1.0 - t_lam)
            past_trigrams[input_ings] = weight(ni, t_lam) + weight(bigram_counter(prev_ing2, mod), 1.0 - t_lam)

        return past_trigrams[input_ings]

#vector utilities
def norm(vec):
    total = 0.0
    for i in range(len(vec)):
        total += vec[i] * vec[i]
    return math.sqrt(total)
def sub_vec(v1,v2):
    ret = []
    for i in range(len(v1)):
        ret += [v1[i]-v2[i]]
    return tuple(ret)
def add_vec(v1,v2):
    ret = []
    for i in range(len(v1)):
        ret += [v1[i]+v2[i]]
    return tuple(ret)
def angle_between(origin, p1, p2):
    v1 = sub_vec(p1, origin)
    v2 = sub_vec(p2, origin)
    return math.acos(dot(v1, v2)/(norm(v1)* norm(v2)))


#callbacks
def speech_callback(input):
    global speech
    speech = input.data.split()
def object_callback(input):
    global objects
    global variance
    global num_objs
    if(len(input.objects) == 1):
        variance = 0.1
    frame = "/base"
    object_frame="/camera_rgb_optical_frame"
    objects = []#[("None", (0,0,0))]
    (translation,rotation) = tfl.lookupTransform(frame, object_frame, rospy.Time(0))
    # (x,y,z) translation (q1, q2, q3, q4) quaternion
    #process into
    # (object_id, (x,y,z))
    for i in range(len(input.objects)):
        cur_obj = input.objects[i].type.key
        cur_loc = input.objects[i].pose.pose.pose.position
        cur_loc_tuple = (cur_loc.x, cur_loc.y, cur_loc.z, 1.0)
        quaternified =dot(cur_loc_tuple, quaternion_matrix(rotation))
        cur_loc_tuple = (translation[0] + quaternified[0],translation[1] + quaternified[1],translation[2] + quaternified[2])
        objects.append((cur_obj, cur_loc_tuple))
    if num_objs == -1:
        num_objs = len(objects)
def truth_callback(input):
    global ground_truth
    ground_truth = input.data




def is_arm_null_gesture(arm_origin, arm_point):
    if (arm_origin == None or arm_point == None):
        return True
    else:
        min_angle = 10.0 #greater than 3.14, so should always be greatest angle
        for obj in objects:
            if angle_between(arm_origin, arm_point, obj[1]) < min_angle:
                min_angle = angle_between(arm_origin, arm_point, obj[1])
        return min_angle > 3.14159/6 or min_angle > angle_between(arm_origin, arm_point, right_foot) or min_angle > angle_between(arm_origin, arm_point, left_foot)
def is_head_null_gesture(origin, point):
    return (origin == None or point == None)

def prob_of_sample(sample):
    return scipy.stats.norm(0.0, math.sqrt(variance)).pdf(sample)


#fills body points from openni data
def fill_points(tfl):
    try:
        global user
        frame = "/base" #"camera_link"
        allFramesString = tfl.getFrameStrings()
        onlyUsers = set([line for line in allFramesString if 'right_elbow_' in line])
        n = len('right_elbow_')
        userIDs = [el[n:] for el in onlyUsers]
        user = ''
        if len(userIDs) > 0:
            mostRecentUID = userIDs[0]
            mostRecentTime = tfl.getLatestCommonTime(frame, 'right_elbow_' + mostRecentUID).to_sec()
            for uid in userIDs:
                compTime = tfl.getLatestCommonTime(frame, 'right_elbow_' + uid).to_sec()
                #rospy.loginfo("Diff time " + str(rospy.get_rostime().to_sec() - compTime))
                if compTime >= mostRecentTime and rospy.get_rostime().to_sec() - compTime < 5:
                    user = uid
                    mostRecentTime = compTime
        global left_arm_origin
        global right_arm_origin
        global head_origin
        global head_point
        global left_arm_point
        global right_arm_point
        global left_foot
        global right_foot
        (to_left_elbow,_) = tfl.lookupTransform(frame,"/left_elbow_" + user, rospy.Time(0))
        (to_right_elbow,_) = tfl.lookupTransform(frame,"/right_elbow_" + user, rospy.Time(0))
        (to_left_hand,_) = tfl.lookupTransform(frame,"/left_hand_" + user, rospy.Time(0))
        (to_right_hand,_) = tfl.lookupTransform(frame,"/right_hand_" + user, rospy.Time(0))
        (right_foot,_) = tfl.lookupTransform(frame, "/right_foot_" + user, rospy.Time(0))
        (left_foot,_) = tfl.lookupTransform(frame, "/left_foot_" + user, rospy.Time(0))
        (to_head,head_rot) = tfl.lookupTransform(frame,"/head_" + user, rospy.Time(0))
        left_arm_origin = to_left_hand
        left_arm_point = add_vec(to_left_hand, sub_vec(to_left_hand, to_left_elbow))
        right_arm_origin = to_right_hand
        right_arm_point = add_vec(to_right_hand, sub_vec(to_right_hand, to_right_elbow))
        head_origin = to_head
        head_temp = dot((0.0,0.0,-1.0,1.0), quaternion_matrix(quaternion_inverse(head_rot)))
        head_point = (head_temp[0] + to_head[0], head_temp[1] + to_head[1], head_temp[2] + to_head[2])
        
        #visualization for testing (verify head vector)
        # marker = Marker()
        # marker.header.frame_id = "camera_link"
        # marker.header.stamp = rospy.Time(0)
        # marker.type = marker.POINTS
        # marker.action = marker.ADD
        # marker.scale.x = 0.2
        # marker.scale.y = 0.2
        # marker.scale.z = 0.2
        # marker.color.a = 1.0
        # p1 = Point(right_arm_origin[0],right_arm_origin[1],right_arm_origin[2])
        # p2 = Point(right_arm_point[0],right_arm_point[1],right_arm_point[2])
        # p3 = Point(left_arm_origin[0],left_arm_origin[1],left_arm_origin[2])
        # p4 = Point(left_arm_point[0],left_arm_point[1],left_arm_point[2])
        # p5 = Point(head_origin[0],head_origin[1],head_origin[2])
        # p6 = Point(head_point[0],head_point[1],head_point[2])
        # marker.points += [p1, p2, p3, p4, p5, p6]
        # pub.publish(marker)

        return True
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        left_arm_point = None
        right_arm_origin = None
        left_arm_origin = None
        right_arm_point = None
        head_point = None
        head_origin = None
        left_foot = None
        right_foot = None
        return False

def baxter_init_response():
    plt.ion()
    plt.figure(figsize=(10,10))
    plt.show()

def plot_respond():
    plt.clf()
    x = []
    for word in state_dist.keys():
        x.append(containers[word])
    plt.bar(range(len(state_dist.keys())), state_dist.values(), align='center')
    plt.xticks(range(len(state_dist.keys())), x, size='small')
    font = {'family' : 'normal','weight' : 'bold','size'   : 25}
    matplotlib.rc('font', **font)
    plt.ylim([0,1.0])
    plt.draw()

def baxter_respond():
    global speech
    global last_obj
    global scnd_to_lat_obj
    most_likely = max(state_dist.iteritems(), key=operator.itemgetter(1))
    if not (is_arm_null_gesture(right_arm_origin, right_arm_point) and \
        is_arm_null_gesture(left_arm_origin, left_arm_point)) or containers[most_likely[0]] in speech:
        if most_likely[1] > 0.9:
            print "SEND PICKUP"
            scnd_to_lat_obj = last_obj
            last_obj = most_likely[0]
            pub = rospy.Publisher('fetch_commands', String, queue_size=0)
            #rospy.init_node('pointer', anonymous=True)
            rate = rospy.Rate(10)
            pub.publish(most_likely[0])
            rate.sleep()
            #print containers[objects[0][0]], speech
        #elif len(objects) == 1 and not (is_arm_null_gesture(right_arm_origin, right_arm_point) and \
         #   is_arm_null_gesture(left_arm_origin, left_arm_point)) or containers[objects[0][0]] in speech:
          #  pub = rospy.Publisher('fetch_commands', String, queue_size=0)
           # print "only one object left"
           # scnd_to_lat_obj = last_obj
           # last_obj = containers[most_likely[0]]
           # #rospy.init_node('pointer', anonymous=True)
           # rate = rospy.Rate(10)
           # pub.publish(most_likely[0])
           # rate.sleep()
    speech = []



def update_model():
    global state_dist
    global speech
    global num_objs
    #if num_objs != len(objects):
    #    for obj in objects:
    #        state_dist[obj[0]] = 1.0/len(state_dist)
    #    num_objs = len(objects)
    #    return
    prev_dist = state_dist
    state_dist = dict()
    #if we have no previous model, set to uniform
    if len(prev_dist.keys()) == 0:
        l = len(objects) *1.0
        for obj in objects: #make sure this is a UID
            prev_dist[obj[0]] = 1.0/l
    for obj in objects:
        print objects
        obj_id = obj[0]
        state_dist[obj_id] = 0.0
        # transition update
        for prev_id in prev_dist.keys():
            if is_arm_null_gesture(right_arm_origin, right_arm_point) and \
             is_arm_null_gesture(left_arm_origin, left_arm_point) \
             and len(speech)==0:
                t= 0.005
            else:
                t = bigram_counter(last_obj, -1)[containers[obj_id]]
                #t = trigram_counter(containers[last_obj], containers[scnd_to_lat_obj], -1)[containers[obj_id]]
                t *= 5
                #t= 0.005
            print "previous object: %s, estimation of object %s: %f" % (last_obj, containers[obj_id], t)
            if prev_id == obj_id:
                state_dist[obj_id] += (1-t)*prev_dist[prev_id]
            else:
                state_dist[obj_id] += t*prev_dist[prev_id]
        # left arm
        if not is_arm_null_gesture(left_arm_origin, left_arm_point):
            l_arm_angle = angle_between(left_arm_origin, left_arm_point, obj[1])
            #if l_arm_angle > 3.14/4:
            #    l_arm_angle = 3.14/2
            state_dist[obj_id] *= prob_of_sample(l_arm_angle)
        #right arm
        if not is_arm_null_gesture(right_arm_origin, right_arm_point):
            r_arm_angle = angle_between(right_arm_origin, right_arm_point, obj[1])
            #if r_arm_angle > 3.14/4:
            #    r_arm_angle = 3.14/2
            state_dist[obj_id] *= prob_of_sample(r_arm_angle)
        #head
        if False and not is_head_null_gesture(head_origin, head_point):
            state_dist[obj_id] *= prob_of_sample(angle_between(head_origin, head_point, obj[1]))
        #speech
        for word in speech:
            if word in vocabulary:
                state_dist[obj_id] *= word_probabilities[obj_id].get(word, eps)
    #normalize
    total = sum(state_dist.values())
    for obj in state_dist.keys():
        state_dist[obj] = state_dist[obj] / total
    global write_speech
    write_speech = speech

def load_dict(filename):
    global word_probabilities
    global vocabulary
    word_probabilities = dict()
    vocabulary = set()
    with open(filename) as f:
        lines = f.read().split('\n')
        for line in lines:
            words = line.split()
            print words
            word_probabilities[words[0]] = dict()
            for i in range(1, len(words)):
                word_probabilities[words[0]][words[i]] = word_probabilities[words[0]].get(words[i], 0.0) + 1.0
                vocabulary.add(words[i])
    for word in word_probabilities.keys():
        total = sum(word_probabilities[word].values())
        for x in word_probabilities[word]:
            word_probabilities[word][x] = word_probabilities[word][x]/ total



def write_output():
    global write_speech
    if storage:
        output = [head_origin, head_point, left_arm_origin, left_arm_point, \
                    right_arm_origin, right_arm_point, left_foot, right_foot,\
                    ground_truth, write_speech, objects, max(state_dist.keys(), key=lambda x: state_dist[x]), time.clock()]
        storage.write(str(output) + "\n")
    write_speech = []
    #objects
    #arms
    #head
    #speech
    #ground truth


def main():
    global speech
    global tfl
    file_reader()
    rospy.init_node('h2r_gesture')
    load_dict(sys.argv[1])
    tfl = tf.TransformListener()
    rospy.Subscriber('speech_recognition', String, speech_callback, queue_size=1)
    rospy.Subscriber('publish_detections_center/blue_labeled_objects', RecognizedObjectArray, object_callback, queue_size=1)
    rospy.Subscriber('current_object', String, truth_callback, queue_size=1)
    rate = rospy.Rate(30.0)
    global storage
    if len(sys.argv) > 2:
        storage = storage = open(sys.argv[2], 'w')


    global pub
    pub = rospy.Publisher("test_marker", Marker)
    marker = Marker()
    marker.header.frame_id = "base" #"camera_link"
    marker.header.stamp = rospy.Time(0)
    marker.type = marker.POINTS
    marker.action = marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    # depth, right left, up down
    #p1 = Point(1.2, 0.07,-0.37) # color bowl
    #p2 = Point(1.2, -0.37, -0.37) #metal bowl
    #p3 = Point(1.5, -0.37, -0.3) #plastic spoon
    #p4 = Point(1.5, 0.07, -0.3) #silver spoon
    #marker.points += [p1,p2,p3,p4]
    baxter_init_response()
    while not rospy.is_shutdown():
        pub.publish(marker)
        fill_points(tfl)
        update_model()
        if not len(state_dist.keys()) == 0:
            for obj in objects:
                marker.points.append(Point(obj[1][0], obj[1][1], obj[1][2]))
            baxter_respond()
            plot_respond()
            write_output()
        rate.sleep()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: rosrun gesture_rec h2r_gesture.py <language model file> <storage file (optional)>"
        sys.exit()
    main()
