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

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

#David's Dependencies
import operator
import collections
import random
import copy

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
scnd_to_lat_obj = "unknown"
third_to_last_obj = "unknown"
storage = None
ground_truth = "unknown"
sent_object = 'unknown'
speech = []
write_speech = []
global state_dist
state_dist = dict()
global objects
objects = []
history =[]
#containers = {"bowl1": "chocolate", "bowl2": "butter", "bowl3": "eggs", "bowl4": "sugar", "bowl5": "vanilla", "bowl6": "flour", "bowl7": "salt", "bowl8": "pepper", "bowl9": "milk", "bowl10": "banana"}
#TEMP HACK
#objects = [("pink_box", (1.4,-0.2,-0.5)), ("purple_cylinder", (1.4, 0.05, -0.5))]
#objects = [("chocolate",(1.2, -0.37, -0.37)), \
#("butter",(1.2, -0.37, -0.37)), \
#("eggs",(1.2, -0.37, -0.37)), \
#("sugar",(1.2, -0.37, -0.37)), \
#("vanilla",(1.2, -0.37, -0.37)), \
#("flour",(1.2, -0.37, -0.37)), \
#("salt",(1.2, -0.37, -0.37)), \
#("pepper",(1.2, -0.37, -0.37)), \
#("milk",(1.2, -0.37, -0.37)), \
#("banana",(1.2, -0.37, -0.37))]
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


#David's global variables
ingredient_file = 'src/no_repeat_numbered.txt'
recipe_list = []
unigram_init = False
uni_counts = collections.Counter()
past_bigrams = {}
past_trigrams = {}
past_4grams = {}
smoothing_coefficient = 0
largest_t = '' #what the language model says is most likely

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
    #smoothing_coefficient = 0.5
#David's utility functions
def normalize(x): #for dictionary vectors
    total = sum(x.values(), 0.0)
    for key in x:
        x[key] /= total

def weight(x, weight):
    cpy = copy.deepcopy(x)
    for key in cpy:
        cpy[key] *= weight
    return cpy

#David's transition functions
def unigram_counter():
    global unigram_init
    global uni_counts
    if unigram_init == False:
        for line in range(0, len(recipe_list)):
            next_ing = recipe_list[line].split(' # ')[1]
            ing_clean = next_ing.split(',')[0]
            if ing_clean in word_probabilities.keys():
                uni_counts[ing_clean] += 1.0
        normalize(uni_counts)
        unigram_init = True
        #print uni_counts
    return uni_counts


def bigram_counter(prev_ing):
    if prev_ing in past_bigrams:
        return past_bigrams[prev_ing]
    else:
        ni = collections.Counter()
        for line in range(0, len(recipe_list)):
            if prev_ing in recipe_list[line - 1]:
                next_ing = recipe_list[line].split(' # ')[1]
                ing_clean = next_ing.split(',')[0]
                #print 'ing clean %s, word_probabilities.keys = %s' %(ing_clean, str(word_probabilities.keys()))
                if ing_clean.replace(' ','_') in word_probabilities.keys():
                    #print ing_clean + 'True!!!!!!!!!!!!!!!!!!!!!!!!!!'
                    ni[ing_clean] += 1.0
        normalize(ni)
        #print  ni
        if not list(ni):
            #return unigram_counter()
            past_bigrams[prev_ing] = unigram_counter()
        else:
            #print 'wooo!'
            past_bigrams[prev_ing] = ni
        return past_bigrams[prev_ing]


def trigram_counter(prev_ing, prev_ing2):
    
    input_ings = prev_ing + ":" + prev_ing2
    if input_ings in past_trigrams:
        return past_trigrams[input_ings]
    else:
        t_lam = smoothing_coefficient
        ni = collections.Counter()

        for line in range(0, len(recipe_list)):
            if prev_ing in recipe_list[line - 2] and prev_ing2 in recipe_list[line - 1]:
                next_ing = recipe_list[line].split(' # ')[1]
                ing_clean = next_ing.split(',')[0]
                if ing_clean.replace(' ','_') in word_probabilities.keys():
                    ni[ing_clean] += 1.0

        normalize(ni)
        if not list(ni):
            #return bigram_counter(prev_ing2)
            past_trigrams[input_ings] = bigram_counter(prev_ing2)
        else:
            #return ni
            past_trigrams[input_ings] = ni
        return past_trigrams[input_ings]

def fourgram_counter(prev_ing, prev_ing2, prev_ing3):

    input_ings = prev_ing + ":" + prev_ing2 + ":" + prev_ing3
    if input_ings in past_4grams:
        return past_4grams[input_ings]
    else:
        f_lam = smoothing_coefficient
        ni = collections.Counter()

        for line in range(0, len(recipe_list)):
            if prev_ing in recipe_list[line - 3] and prev_ing2 in recipe_list[line - 2] and prev_ing3 in recipe_list[line - 1]:
                    next_ing = recipe_list[line].split(' # ')[1]
                    ing_clean = next_ing.split(',')[0]
                    if ing_clean in word_probabilities.keys() and ing_clean not in last_obj and ing_clean not in scnd_to_lat_obj and ing_clean not in third_to_last_obj:
                        ni[ing_clean] += 1.0

        normalize(ni)
        if not list(ni):
            #return trigram_counter(prev_ing2, prev_ing3)
            past_4grams[input_ings] = trigram_counter(prev_ing2, prev_ing3)
        else:
            #return ni
            past_4grams[input_ings] = ni
        return past_4grams[input_ings]


#callbacks
def speech_callback(input):
    global speech
    speech = input.data.split()
    #baxter_respond()
def object_callback(input):
    pass
def truth_callback(input):
    global ground_truth
    ground_truth = input.data


def update_history():   
    global last_obj
    global scnd_to_lat_obj
    global third_to_last_obj

    if ground_truth != last_obj:
        third_to_last_obj = scnd_to_lat_obj
        scnd_to_lat_obj = last_obj
        last_obj = ground_truth
        reset_history()



def prob_of_sample(sample):
    return scipy.stats.norm(0.0, math.sqrt(variance)).pdf(sample)


def baxter_init_response():
    plt.ion()
    plt.figure(figsize=(10,10))
    plt.show()

def plot_respond():
    plt.clf()
    x = []
    for word in state_dist.keys():
        x.append(word)
    plt.bar(range(len(state_dist.keys())), state_dist.values(), align='center')
    plt.xticks(range(len(state_dist.keys())), x, size='small', rotation='vertical')
    font = {'family' : 'normal','weight' : 'bold','size'   : 25}
    matplotlib.rc('font', **font)
    plt.ylim([0,1.0])
    plt.draw()

def baxter_respond():
    global speech
    global history
    global sent_object

    print 'obj 1: %s, obj 2: %s, obj 3: %s' % (last_obj, scnd_to_lat_obj, third_to_last_obj)

    most_likely = sorted(state_dist.iteritems(), key=operator.itemgetter(1))
    if speech:
            pub = rospy.Publisher('fetch_commands', String, queue_size=0)
            #rospy.init_node('pointer', anonymous=True)
            rate = rospy.Rate(10)
            best_ing = -1
            pub.publish(most_likely[best_ing][0])
            print "SENT OBJECT:" + most_likely[best_ing][0]
            #for obj in objects:
            #    #print obj[0], most_likely[0]
            #    if obj[0] == most_likely[0]:
            #        objects.remove(obj)
            #del state_dist[most_likely[0]]
            rate.sleep()
            history.append(ground_truth)
    speech = []






def update_model():
    global state_dist
    global speech
    global num_objs
    global largest_t
    global speech
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
        obj_id = obj[0]
        state_dist[obj_id] = 0.0
        # transition update
        for prev_id in prev_dist.keys():
            if len(speech)==0 and False:
                t = 0.005
            else:
                #t= 0.005
                #t = max(0.0001, unigram_counter()[obj_id.replace('_',' ')])
                #t = unigram_counter()[obj_id.replace('_',' ')]
                #cur_dist = bigram_counter(last_obj)
                #print cur_dist
                t = max(0.0001, bigram_counter(ground_truth)[obj_id.replace('_',' ')])
                #old_t = largest_t
                #largest_t = cur_dist.most_common()[0][0]
                #if old_t != largest_t:
                #    old_t = largest_t
                #    print largest_t
                #t = max(0.0001, trigram_counter(scnd_to_lat_obj, last_obj)[obj_id.replace('_', ' ')])
                #t = max(0.0001, fourgram_counter(last_obj, scnd_to_lat_obj, third_to_last_obj)[obj_id.replace('_',' ')])
                #if 'brown sugar' in speech:
                #    print 'brown sugar %f' % t
                #if 'white sugar' in speech:
                #    print 'white sugar %f' % t
                #if 'artichokes' in obj_id:
                   # print 't for %s is bigram(%s) = %f' % (obj_id, last_obj, t)
                #t *= 10
            #print 'last_obj %s, obj_id %s, t %f' % (last_obj, obj_id, t)    
            #t = bigram_counter(last_obj)[obj_id.replace('_', ' ')]
            #t = trigram_counter(last_obj, scnd_to_lat_obj)[obj_id.replace('_', ' ')]
            #t *= 10
            #t = 0.05
            #print "previous object: %s, estimation of object %s: %f" % (last_obj, containers[obj_id], t)
            if prev_id == obj_id:
                state_dist[obj_id] += (1-t)*prev_dist[prev_id]
            else:
                state_dist[obj_id] += t*prev_dist[prev_id]
    
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

    

def reset_history():
    if ground_truth == 'sesame_seeds':
        global last_obj
        global scnd_to_lat_obj
        global third_to_last_obj
        global history

        last_obj = 'unknown'
        scnd_to_lat_obj = 'unknown'
        third_to_last_obj  = 'unknown'
        history = []

def load_dict(filename):
    global word_probabilities
    global vocabulary
    global objects
    word_probabilities = dict()
    vocabulary = set()
    with open(filename) as f:
        lines = f.read().split('\n')
        for line in lines:
            words = line.split()
            #print words
            objects.append((words[0], (1.2, -0.37, -0.37)))
            if words[0] not in word_probabilities:
                word_probabilities[words[0]] = dict()
            for i in range(1, len(words)):
                word_probabilities[words[0]][words[i]] = word_probabilities[words[0]].get(words[i], 0.0) + 1.0
                vocabulary.add(words[i])
    for word in word_probabilities.keys():
        total = sum(word_probabilities[word].values())
        for x in word_probabilities[word]:
            word_probabilities[word][x] = word_probabilities[word][x]/ total

    #print word_probabilities



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
    rospy.Subscriber('ground_truth', String, truth_callback, queue_size=1)
    rate = rospy.Rate(30.0)
    global storage
    if len(sys.argv) > 2:
        storage = storage = open(sys.argv[2], 'w')


    global pub
    pub = rospy.Publisher("test_marker", Marker, queue_size=1)
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
        update_model()
        if not len(state_dist.keys()) == 0:
            for obj in objects:
                #marker.points.append(Point(obj[1][0], obj[1][1], obj[1][2]))
                pass
            baxter_respond()
            plot_respond()
            update_history()
        for i in range(0, 10):
            #update_model()
            pass
        rate.sleep()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: rosrun gesture_rec h2r_gesture.py <language model file> <storage file (optional)>"
        sys.exit()
    main()
