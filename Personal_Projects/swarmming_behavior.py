import pygame as pg
import random
pg.init()

Width, Height = 500,500
FPS = 10
WIN = pg.display.set_mode((Width,Height))
pg.display.set_caption("Swarming")
Black = (0,0,0)
White = (255,255,255)
Prey_number = 40
Predator_number = 1

class Ball:
    Max_Vel = 5
    
    def __init__ (self, x,y, radius, Color):
        self.Color = Color
        self.x = self.original_x =  x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel =  random.randint(-1*self.Max_Vel,self.Max_Vel)
        self.y_vel = random.randint(-1*self.Max_Vel,self.Max_Vel)

    def draw(self, win):
        pg.draw.circle(win, self.Color,(self.x, self.y), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel
    
    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = 0
        self.x_vel *= -1


    def draw(self, win):
        pg.draw.circle(win, self.Color,(self.x, self.y), self.radius)


def draw(Win, prey, predators):
    Win.fill(Black)
    for bird in prey:
        bird.draw(Win)
    for bird in predators:
        bird.draw(Win)
    
    pg.display.update()



def handle_interactions(prey, predators):
    for pidgeon in prey:
        for bird in prey:
            pidgeon.x_vel += (bird.x - pidgeon.x)//Prey_number
            pidgeon.y_vel += (bird.y - pidgeon.y)//Prey_number
        for bird in predators:
            pidgeon.x_vel += ( pidgeon.x- bird.x )/Predator_number + 10/pidgeon.x - 10/(Width- pidgeon.x)
            pidgeon.y_vel += ( pidgeon.y- bird.y)/Predator_number +  10/pidgeon.y - 10/(Height- pidgeon.y)

        pidgeon.x_vel *= 1/((pidgeon.x_vel **2 + pidgeon.y_vel**2)**.5) 
        pidgeon.y_vel *= 1/((pidgeon.x_vel **2 + pidgeon.y_vel**2)**.5) 





    for hawk in predators:
        for bird in prey:
            hawk.x_vel = (bird.x - hawk.x ) //Prey_number
            hawk.y_vel = (bird.y - hawk.y)  //Prey_number
        







def main():
    run = True
    clock = pg.time.Clock()
    prey = []
    for i in range(0,Prey_number):
        prey.append(Ball(random.randint(1,Width), random.randint(1,Height), 2, 'green'))





    predators=[]

    for i in range(0,Predator_number):
        predators.append( Ball(random.randint(1,Width), random.randint(1,Height), 2, 'red'))



    


    while run:
        clock.tick(FPS)
        draw(WIN,prey,predators)
   
    


       

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                break
            
        
        handle_interactions(prey,predators)
        for bird in prey:
            bird.move()

        for bird in predators:
            bird.move()

       

        
    


        

  
        
        


if __name__ == '__main__':
    main()



