#!/usr/bin/env python

import time
import sys



import math
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


#David's Dependencies
import operator
import collections
import random
import copy
import csv

responses = [] #list of all commands


#falafel
falafel_ingredients = ('chickpeas', \
	'chickpeas', \
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

#puff pastry
pp_ingredients = ('puff_pastry', \
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

#chow mein
chow_mein_ingredients = ('noodles', \
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

#spanish rice
spanish_rice_ingredients = ('olive_oil', \
	'olive_oil', \
	'garlic', \
	'garlic', \
	'bell_pepper', \
	'bell_pepper', \
	'shallots', \
	'shallots', \
	'long_grain_rice', \
	'long_grain_rice', \
	'tomato_sauce', \
	'tomato_sauce', \
	'chicken_stock', \
	'chicken_stock', \
	'salt', \
	'salt', \
	'black_pepper', \
	'black_pepper', \
	'oregano', \
	'oregano' )

#beef curry
beef_curry_ingredients = ('beef', \
	'beef', \
	'onion', \
	'onion', \
	'garlic', \
	'garlic', \
	'ginger', \
	'ginger', \
	'curry_powder', \
	'curry_powder', \
	'cayenne', \
	'cayenne', \
	'coconut_milk', \
	'coconut_milk', \
	'soy_sauce', \
	'soy_sauce', \
	'fish_sauce', \
	'fish_sauce', \
	'bell_pepper', \
	'bell_pepper', \
	'cherry_tomatos', \
	'cherry_tomatos', \
	'shallots', \
	'shallots' )

recipe_set = [falafel_ingredients, pp_ingredients, chow_mein_ingredients, spanish_rice_ingredients, beef_curry_ingredients]

csv_set = ['Batch_falafel.csv', 'Batch_puff_pastry.csv', 'Batch_chow_mein.csv', 'Batch_spanish_rice.csv', 'Batch_beef_curry.csv']


global state_dist
state_dist = dict()
global objects
objects = []
history =[]

global t
t = 0.005
global variance
variance = 0.4
global word_probabilities
global vocabulary
global eps
eps = 0.0001


#David's global variables
ingredient_file = 'no_repeat_numbered.txt'
recipe_list = []
unigram_init = False
uni_counts = collections.Counter()
past_bigrams = {}
past_trigrams = {}
past_4grams = {}
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


def load_dict(filename):
    global word_probabilities
    global vocabulary
    global objects
    word_probabilities = dict()
    vocabulary = set()
    ings = []
    with open(filename) as f:
        lines = f.read().split('\n')

        for line in lines:
            words = line.split()
            ings.append(words[0])
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
    print sorted(ings), len(ings)
    #print word_probabilities

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
                    if ing_clean in word_probabilities.keys():
                        ni[ing_clean] += 1.0

        normalize(ni)
        if not list(ni):
            #return trigram_counter(prev_ing2, prev_ing3)
            past_4grams[input_ings] = trigram_counter(prev_ing2, prev_ing3)
        else:
            #return ni
            past_4grams[input_ings] = ni
        return past_4grams[input_ings]

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

def csv_parser(csv_file):
	global responses
	responses = []
	with open(csv_file, 'rb') as csvfile:
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

def update_model(speech, ground_truth, gesture):
    global state_dist
    global objects
    #print history[-1]

    prev_dist = state_dist
    state_dist = dict()
    #if we have no previous model, set to uniform
    if len(prev_dist.keys()) == 0:
        l = len(objects) *1.0
        for obj in objects: #make sure this is a UID
            prev_dist[obj[0]] = 1.0/l

    if gesture:
    	objects_backup = copy.deepcopy(objects)
    	new_objects = []
    	new_objects.append(objects[objects.index((ground_truth,(1.2,-0.37,-0.37)))])                                       
    	objects.remove((ground_truth,(1.2,-0.37,-0.37)))
    	for i in range(0, 10):
    		random_ingredient = random.choice(objects)
    		new_objects.append(random_ingredient)
    		objects.remove(random_ingredient)
    	objects = new_objects

    for obj in objects:
        obj_id = obj[0]
        state_dist[obj_id] = 0.0
        # transition update
        for prev_id in prev_dist.keys():
            #t = 0.005
            #t = max(0.005, unigram_counter()[obj_id.replace('_',' ')])
            t = max(0.005, bigram_counter(history[-1])[obj_id.replace('_',' ')])
            #t = max(0.005, trigram_counter(history[-2], history[-1])[obj_id.replace('_', ' ')])
            #t = max(0.005, fourgram_counter(history[-3], history[-2], history[-1])[obj_id.replace('_',' ')])

            if prev_id == obj_id:
                state_dist[obj_id] += (1-t)*prev_dist[prev_id]
            else:
                state_dist[obj_id] += t*prev_dist[prev_id]
    
        #speech
        for word in speech.split():
        	#print word
        	if word in vocabulary:
        		#print 'test'
        		state_dist[obj_id] *= word_probabilities[obj_id].get(word, eps) 
    #normalize
    total = sum(state_dist.values())
    for obj in state_dist.keys():
        state_dist[obj] = state_dist[obj] / total

    #plot_respond()

    if gesture:
    	objects = objects_backup

    #return most likely item
    sorted_state_dist = max(state_dist.iteritems(), key=operator.itemgetter(1))
    return sorted_state_dist[0]
    #return random.choice(state_dist.keys())



def talker():
	global cur_ing
	global history
	#print len(ingredients),  len(responses[1])

	gesture = True

	total = 0.0
	hit = 0.0
	for recipe in range(0, len(recipe_set)):
		csv_parser(csv_set[recipe])
		ingredients = recipe_set[recipe]
		print 'recipe: ' + csv_set[recipe]
		for i in range(0, len(responses)):
			print 'user #%d' % i
			if i%2 == 1 or True:
				history = ['unknown', 'unknown', 'unknown']
				for j in range(0, len(responses[i])):
					if j%2 == 1:
						total += 1
						clean_request = responses[i][j].replace('.','').replace(',','').replace('?','').lower()
						request_list = clean_request.split()
						clean_request = ' '.join(request_list)
						prediction = update_model(clean_request, ingredients[j], gesture)
						print 'prediction: ' + prediction + ' actual: ' + ingredients[j] + ': ' + ' history: ' + history[-1] + ' request: ' + clean_request
						#print state_dist
						if ingredients[j] == prediction:
							#print "yay!!"
							hit += 1
						history.append(ingredients[j])
						#reset state_dis
						for k in range(0, 5):
							update_model('', 'n/a', False)

		print 'accuracy = %f' % (hit/total)
	print 'final accuracy = %f. sample size: %f' % (hit/total, total)




def main():
    global speech
    random.seed(3797)
    file_reader() 
    load_dict(sys.argv[1])
    global storage
    if len(sys.argv) > 2:
        storage = storage = open(sys.argv[2], 'w')

    #sets up plot
    #baxter_init_response()

    #update_model('')

    #goes through AMT data one by one, feeding into model and checking result
    #calls update model
    talker()



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: rosrun gesture_rec h2r_gesture.py <language model file> <storage file (optional)>"
        sys.exit()
    main()
