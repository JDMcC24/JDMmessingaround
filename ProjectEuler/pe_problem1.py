"""If we list all the natural numbers below 10
 that are multiples of 3 or 5 , we get 3,5,6and 9. The sum of these multiples is 23.
Find the sum of all the multiples of 
 3 or 5 below 1000 ."""


#Obvious Method
multiples = []
for i in range(1000):
    if i%3 == 0 or i%5 == 0:
        multiples.append(i)
print(sum(multiples))



#Simplier Method
multiples2 = set([])
for i in range(334):
    multiples2.add(i*3)

for i in range(200):
    multiples2.add(i*5)

print(sum(list(multiples2)))
