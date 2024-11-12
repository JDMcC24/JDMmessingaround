from my_functions import *


def count_digits(n):
    #Counts number of digits in n
    n_string = str(n)
    return len(n_string)

def Fibonacci_list(n):
    #creates a list of the first n > 2 Fibonacci numbers
    F_list = [1, 1] 
    while len(F_list) < n:
        F_list.append( F_list[-1] + F_list[-2])
    return F_list

F_list=[1,1]
limit = 1000
while count_digits(F_list[-1]) < limit:
    F_list.append( F_list[-1] + F_list[-2])

print(len(F_list))