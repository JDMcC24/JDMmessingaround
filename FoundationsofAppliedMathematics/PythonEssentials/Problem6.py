def pig_latin(a):
    vowels = {"a",'e','i','o','u','y'}
    if a[0] in vowels:
        a = a+"hay"
    else:
        a = a+ a[0]+"ay"
        a = a[1:]
    return a

print(pig_latin("elephant"))
print(pig_latin("hello"))

