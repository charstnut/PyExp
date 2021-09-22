# This is a notebook for testing and reference
# %%
import IPython
import lmfit as fit
import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
import uproot
from lmfit import Parameters, minimize
from matplotlib import pyplot as plt
from scipy import integrate, interpolate, linalg, odr, optimize, signal, stats

IPython.get_ipython().magic('matplotlib auto')

# %%
print('this is a string' + ' this is another string')
print('you can multiply strings' * 2)
print('a')
print('b')

# %%
# make an array
a = [[1, 2, 3], [4, 5, 6]]
a.append([2, 3, 4])  # append additional lists
b = np.zeros((2, 3), int)
print(a, b, type(a), type(b))
print('---------------')

# add two lists together, note b is np.ndarray not list so conversion is needed
a = np.asarray(a, int)
c = np.array([[6, 7, 8], [7, 8, 9]])
c = c.T  # translate an array by switching its dimensions
print(c)
print('---------------')
# append ndarrays, note axis = 0 means the along the rows, axis = 1 means along the columns, etc.
d = np.append(a, b, axis=0)
print(d)
print('---------------')
# concatenate also works, for a sequence of arrays
e = np.concatenate((a, b.T, c), axis=1)

print(e)
print('---------------')

# we can also perform calculations on arrays directly
e / 30
np.divide(e, 30) # can also use this for element division in arrays

# %%
# element-wise operations
array1 = np.arange(2, 10001, 2)
array2 = array1 - 1
array3 = array1 + 1
# array1 = (2,4,6)
# array2 = (1,3,5)
np.prod(array1/array2) * np.prod(array1/array3) * \
    2  # Wallis product: should be Pi

# %%
# for pandas
# Series is just 1-d data tables
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s

# %%
data = {'one': [1, 2, 3, 4],
        'two': [4, 5, 6, 7]}  # data can be generated as dictionaries
frame = pd.DataFrame(data, index=['a', 'b', 'c', 'd'])
frame

# %%
# and we can add or del columns
frame['three'] = frame['one'] > 2
frame
del frame['two']
frame

# %%
data2 = np.zeros((2, 3))
# passing data as numpy arrays (2D)
frame2 = pd.DataFrame(data2, index=['a', 'b'], columns=['one', 'two', 'three'])
frame2

# %%
data3 = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
         'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
# with Series or dict lists dataframe can do simple sorting during creation
frame3 = pd.DataFrame(data3, index=['d', 'b', 'a'])
frame3

# %%
# plot a line, implicitly creating a subplot(111)
plt.plot([1, 2, 3])
# now create a subplot which represents the top plot of a grid
# with 2 rows and 1 column. Since this subplot will overlap the
# first, the plot (and its axes) previously created, will be removed
plt.subplot(211)
plt.plot(range(12))
plt.subplot(212, facecolor='y')  # creates 2nd subplot with yellow background

# %%
# Load file
A = np.genfromtxt('./data/va.txt')
S = np.genfromtxt('./data/vs.txt')

plt.figure(figsize=(10, 6))
plt.hist(A)

plt.figure(figsize=(10, 6))
plt.hist(S)

print(np.mean(A), np.mean(S))

sa2 = len(A)*np.std(A)**2
ss2 = len(S)*np.std(S)**2
sa = sum(A*S) - sum(A)*sum(S)/len(A)
print(sa2)
print(ss2)
print(sa)
print(sa/np.sqrt(sa2*ss2))

pool = np.vstack((A, S))
admitted = pool[:, np.logical_or(A > 0.95, S > 0.95)]

print(sum(admitted[0]), sum(admitted[1]),
      np.mean(admitted[0]), np.mean(admitted[1]))

sa2 = len(admitted[0])*np.std(admitted[0])**2
ss2 = len(admitted[1])*np.std(admitted[1])**2
sa = sum(admitted[0]*admitted[1]) - sum(admitted[0]) * \
    sum(admitted[1])/len(admitted[0])
print(sa2)
print(ss2)
print(sa)
print(sa/np.sqrt(sa2*ss2))
