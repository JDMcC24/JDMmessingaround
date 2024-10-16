import pygame as pg
pg.init()
import random

WIDTH , HEIGHT = 800 , 800

WIN = pg.display.set_mode((WIDTH,HEIGHT))
pg.display.set_caption("Snake")
FPS = 60
SCORE_FONT = pg.font.SysFont("comicsans", 50)

FPS = 10
Black = (0,0,0)
White = (255,255,255)
BLOCK_SIZE = 50



class Snake:
    def __init__(self):
        self.x, self.y = BLOCK_SIZE, BLOCK_SIZE
        self.x_vel = 1
        self.y_vel = 0
        self.body = [pg.Rect(self.x- BLOCK_SIZE, self.y, BLOCK_SIZE, BLOCK_SIZE)]
        self.head = pg.Rect(self.x  , self.y, BLOCK_SIZE, BLOCK_SIZE)
        self.dead = False

    def update(self):
        global apple
        for square in self.body[0:len(self.body)-1]:
            if self.head.x == square.x and self.head.y == square.y:
                self.dead = True
            if self.head.x  not in range(0, WIDTH) or self.head.y not in range(0,HEIGHT):
                self.dead = True

        if self.dead:
            self.x, self.y = BLOCK_SIZE, BLOCK_SIZE
            self.x_vel = 1
            self.y_vel = 0
            self.body = [pg.Rect(self.x- BLOCK_SIZE, self.y, BLOCK_SIZE, BLOCK_SIZE)]
            self.head = pg.Rect(self.x  , self.y, BLOCK_SIZE, BLOCK_SIZE)





        self.body.append(self.head)
        for i in range(len(self.body)-1):
            self.body[i].x, self.body[i].y = self.body[i+1].x, self.body[i+1].y
        self.head.x += self.x_vel * BLOCK_SIZE
        self.head.y += self.y_vel * BLOCK_SIZE
        self.body.remove(self.head)

class Apple:
    def __init__(self):
        self.x = random.randint(0,WIDTH//BLOCK_SIZE) * BLOCK_SIZE
        self.y = random.randint(0,WIDTH//BLOCK_SIZE) * BLOCK_SIZE
        self.block = pg.Rect(self.x, self.y, BLOCK_SIZE, BLOCK_SIZE)

    def reset(self):
        self.x = random.randint(0,WIDTH//BLOCK_SIZE) * BLOCK_SIZE
        self.y = random.randint(0,WIDTH//BLOCK_SIZE) * BLOCK_SIZE
        self.block = pg.Rect(self.x, self.y, BLOCK_SIZE, BLOCK_SIZE)

snake = Snake()
apple = Apple()

def draw():

    for x in range(0,WIDTH, BLOCK_SIZE):
        for y in range(0, HEIGHT, BLOCK_SIZE):
            rect = pg.Rect(x,y, BLOCK_SIZE, BLOCK_SIZE)
            pg.draw.rect(WIN, White , rect,1)
    # WIN.fill(White)

    pg.draw.rect(WIN, "green", snake.head)
    for square in snake.body:
        pg.draw.rect(WIN, "green", square)

    pg.draw.rect(WIN, "red", apple.block)
    
    

    pg.display.update()






def main():
    run = True
    clock = pg.time.Clock()
    Score = 0

    while run:
        clock.tick(FPS)
        draw()
        if snake.dead:
            end_text = "Game Over. Final score = " + str(Score)
            text = SCORE_FONT.render(end_text,1, "green")
            WIN.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height() //2 - BLOCK_SIZE ))
            apple.reset()
            pg.display.update()
            pg.time.delay(5000)
            
            snake.dead = False


        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_DOWN and snake.y_vel !=-1:
                    snake.y_vel = 1
                    snake.x_vel = 0
                if event.key == pg.K_UP and snake.y_vel !=1:
                    snake.y_vel = -1
                    snake.x_vel = 0
                if event.key == pg.K_RIGHT and snake.x_vel !=-1:
                    snake.y_vel = 0
                    snake.x_vel = 1
                if event.key == pg.K_LEFT and snake.x_vel !=1:
                    snake.y_vel = 0
                    snake.x_vel = -1
            

            if event.type == pg.QUIT:
                run = False
                break
        
        snake.update()

        if snake.head.x == apple.x and snake.head.y == apple.y:
            Score +=1
            apple.reset()
            snake.body.append(pg.Rect(snake.head.x, snake.head.y, BLOCK_SIZE, BLOCK_SIZE))
            
            


            
        

        WIN.fill(Black)
        keys = pg.key.get_pressed()

main()