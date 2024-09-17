import numpy as np
data=np.load('area.npy', allow_pickle=True)
print(data[3])
from scipy.stats import norm
import matplotlib.pyplot as plt
for data_ in data:
    mu, std = norm.fit(data_)
    print(min((data_)), max(data_), mu, std, len(data_))
    pdf = norm.pdf(data_, mu, std)
    plt.hist(data_, density=True, alpha=0.6, bins=30)
    #plt.plot(data_, pdf, 'r', linewidth=2)
    plt.show()