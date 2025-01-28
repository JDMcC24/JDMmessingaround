def prob2():
    x = 1       #int
    y = x
    y = 2
    print("does x =y?", x == y)      # False, so Immutable

    mystr1 = "oops I did it" #str
    mystr2 = mystr1
    mystr2 ="oops I did it" + "again..."
    print(  "does mystr1 = mystr2?", mystr1 == mystr2) #False, so Immutable

    mylist1 = [1,2,3,4,5]   #lists
    mylist2 = mylist1
    mylist2[1] = 0
    print(  "does mylist1 = mylist2?", mylist1 == mylist2) #True, so Mutable

    myset1 = {"Bad Religion", "Anti-flag" , "Minor Threat", "Operation Ivy", "NOFX", "Dead Kennedeys", "Rancid"} #Sets
    myset2 = myset1
    myset2.add("Reel Big Fish")
    print(  "does myset1 = myset2?", myset1 == myset2) #True, so Mutable

    mytuple1 = (1,2,3,4)   #Tuples
    mytuple2 = mytuple1
    mytuple2 += (1,)

    print(  "does mytuple1 = mytuple2?", mytuple1 == mytuple2) #False, so Immutable

    return (x == y, mystr1 == mystr2, mylist1 == mylist2, myset1 == myset2, mytuple1 == mytuple2 )