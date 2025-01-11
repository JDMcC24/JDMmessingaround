def list_ops():
    mylist = ["bear", "ant", "cat","dog"]
    mylist.append("eagle")
    mylist.pop(1)
    mylist = mylist[::-1]
    i = mylist.index("eagle")
    mylist[i] = "hawk"
    mylist[-1] = mylist[-1]+"hunter"
    print(mylist)


list_ops()

