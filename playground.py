import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import halfnorm, expon, uniform, chi


def generate_items_halfnormal(amount):
    data = {}
    for i in range(amount):
        data[i] = halfnorm.pdf(i, scale=(amount/3)) 
    return data                                    

def generate_exponential(amount): #Generate a popular playlist
    data = {}
    for i in range(amount):
        data[i] = expon.pdf(i, scale=(amount/7)) 
    return data

def generate_exponential_inv(amount): #Generates a "niche" (unpopuar) playlist 
    data = {}
    for i in range(amount):
        data[amount - 1 - i] = expon.pdf(i, scale=(amount/7)) 
    return data


def generate_uniform(amount): #Generates an uniform distribution among the items
    data = {}
    for i in range(amount):
        data[i] = uniform.pdf(i, scale=amount) 
    return data

def generate_chi(amount): #Generates an uniform distribution among the items
    data = {}
    df = 78
    for i in range(amount):
        data[i] = chi.pdf(i, df, scale=amount) 
        print(data[i])
    return data


def get_cumulative_prob(y):

    cum_prob = {}
    cum_prob[0] = y[0]

    for i in range(1, 300):
        cum_prob[i] = cum_prob[i-1] + y[i]
    return cum_prob

def get_playlists(cum_prob):

    playlist = [] #empty toy playlist



    #create random playlist (like this bc they dont add to 1)
    while len(playlist) < 100:

        prob = np.random.random()
        #print(prob)
        finished = False
        i = 0
        while i < 300 and finished == False:
            if prob < cum_prob[i] and i not in playlist:
                playlist.append(i)
                finished = True
            i += 1

    playlist.sort()
    #print(playlist)
    return playlist


data = generate_exponential(300)
#data = generate_exponential_inv(300)
#data = generate_uniform(300)
lists = sorted(data.items())
x, y = zip(*lists)
cum_prob = get_cumulative_prob(y)
playlist = get_playlists(cum_prob)
plt.plot(x, y)
plt.show()

"""If we wanted to make 3000 playlists and the "popular" playlists are 3 times 
more popular than balanced playlists, and so on the numbers are 2076 popular, 
693 balanced, and 231 unpopular
Other option is probabilities x = 69.2307692308%; y = 23.0769230769%; z = 7.69230769231%
Other option is to randomly decide the type of the playlists"""

playlist_list = []
for i in range(3000):
    type = random.randint(0,2)
    if type == 1:
        playlist[i] = generate_exponential(300)
    elif type == 2:
        playlist[i] = generate_exponential_inv(300)
    else:
        playlist[i] = generate_uniform(300)