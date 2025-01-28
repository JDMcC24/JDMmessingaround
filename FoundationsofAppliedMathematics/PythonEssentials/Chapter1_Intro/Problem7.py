from time import time

def is_palindrome(num):
    num_str = str(num)
    return num_str == num_str[::-1]


# starttime = time()

# #This works but is wildly inefficient
# palindromes = []
# for i in range(100,1000):
#     for j in range(i,1000):
#         if is_palindrome(i*j):
#             palindromes.append(i*j)
# print(max(palindromes), time() - starttime)



#Better method
starttime = time()
p = 0
for i in range(100,1000):
    for j in range(i,1000):
        if is_palindrome(i*j) and i*j > p:
            p = i*j

print(p, time() - starttime)