import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import halfnorm


def generate_items_halfnormal(amount):
    data = {}
    for i in range(amount):
        data[i] = halfnorm.pdf(i, scale=(amount/3)) #THE PROBABILITIES DONT ADD TO 1, NOT BEING ABLE TO CHOOSE
    return data                                     #FROM THE LIST TO CREATE "PLAYLISTS"

data = generate_items_halfnormal(300)
lists = sorted(data.items())
x, y = zip(*lists)
"""plt.plot(x, y)
plt.show()"""


cum_prob = {}
cum_prob[0] = y[0]

for i in range(1, 300):
    cum_prob[i] = cum_prob[i-1] + y[i]
    #print(cum_prob[i])

playlist = [] #empty toy playlist


#create random playlist (like this bc they dont add to 1)
while len(playlist) < 100:

    prob = np.random.random()
    print(prob)
    finished = False
    i = 0
    while i < 300 and finished == False:
        if prob < cum_prob[i] and i not in playlist:
            playlist.append(i)
            finished = True
        i += 1

playlist.sort()
print(playlist)

#list = np.random.choice(x, 100,p=y)
#print(list)


#1. Generate playlists (ones that are popular based, others that are not)
#2. Calculate popularity bias?
#3. Train a model with the data (predict next song in the playlist)
#4. Calculate uncertainty
#5. Relation between popularity and uncertainty?


# fig, ax = plt.subplots(1, 1)
# r = halfnorm(scale=100).rvs(size=1000)
# print(halfnorm.pdf(r[0]), r[0])
# ax.hist(r, density=True, histtype='stepfilled', alpha=0.5)
# ax.legend(loc='best', frameon=False)
# plt.show()




# fig, ax = plt.subplots(1, 1)
# a = 4

# mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')
# x = np.linspace(skewnorm.ppf(0.01, a),
#                 skewnorm.ppf(0.99, a), 100)
# ax.plot(x, skewnorm.pdf(x, a),
#         'r-', lw=5, alpha=0.6, label='skewnorm pdf')

# rv = skewnorm(a)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# vals = skewnorm.ppf([0.001, 0.5, 0.999], a)
# np.allclose([0.001, 0.5, 0.999], skewnorm.cdf(vals, a))

# r = skewnorm.rvs(a, size=1000)

# ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
# ax.legend(loc='best', frameon=False)
# print(r)
# plt.show()

# x = np.linspace(-50,110,100)
# pdf_result = skewnorm.pdf(x, 4,loc=0,scale=30)

# r = skewnorm.rvs(4, size=1000)
# print(r)

# plt.plot(x,pdf_result *100)
# plt.xlabel('x-data')
# plt.ylabel('pdf_value')
# plt.title("PDF of a Normal Distribution with mean=0 and sigma=1")
# plt.show()

# from scipy.stats import halfnorm
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)

# mean, var, skew, kurt = halfnorm.stats(moments='mvsk')
# x = np.linspace(halfnorm.ppf(0.01),
#                 halfnorm.ppf(0.99), 100)

# ax.plot(x, halfnorm.pdf(x),
#        'r-', lw=5, alpha=0.6, label='halfnorm pdf')

# rv = halfnorm
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# vals = halfnorm.ppf([0.001, 0.5, 0.999])
# print(np.allclose([0.001, 0.5, 0.999], halfnorm.cdf(vals)))

# r = halfnorm.rvs(size=1000)

# ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
# ax.legend(loc='best', frameon=False)
# plt.show()