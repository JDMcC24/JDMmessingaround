from random import randint

def roll(s):
    if s <= 6:
        return randint(1,6)
    d1 = randint(1,6)
    d2 = randint(1,6)
    return d1+ d2



# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box"""
    remaining = list(range(1,10))
    if len(sys.argv) == 3: 
        cheat = False;
        t = 0
        
        tlimit = float(sys.argv[2])
        while t <= tlimit:                       #Initiate game
            print("Numbers left:" + str(remaining))         #List of numbers left
            print( 'Seconds left: ' + str(round(tlimit - t,2)))               #Time remaining
            r = roll(sum(remaining))
            box.isvalid(r, remaining)
            if box.isvalid(r, remaining) == False:
                print("You had" + str(remaining) + "left, but you rolled " + str(r)+ ". Yer luck's run out mate. You lose. " )
                break
            print('Roll:'+ str(r))
            start = time.time()                      #Start the round's timer
            choice = input("Which ones you removing mate?")
            choice = box.parse_input(choice, remaining)
            end = time.time()                        #End the round's timer

            t += end-start                           #total time used
            if sum(choice) != r:
                cheat = True

            for i in choice:                         #Checking that choice was with the rules
                if i in remaining == False:
                    cheat = True
                    
            if cheat == True:
                print("Oi!! What are you trying to pull you doorknob? Bugger off, I'm not playing with a cheat")
                break 

            for i in choice:                        #Removing the choosen numbers
                remaining.remove(i)

            if sum(remaining) == 0:                 #Win conditions
                print('Well played lad, you win!')
                break  
        if t > tlimit:
            print("Yer time's up lad, You lose.")
    

    else:                                           #Not enough arguments given
        print("Oi you doorknob, you never told me your name or how long we are playing. ")