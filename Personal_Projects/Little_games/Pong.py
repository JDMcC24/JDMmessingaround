import pygame as pg
pg.init()



WIDTH, HEIGHT = 700, 500
Paddle_WIDTH , Paddle_HEIGHT = 20, 100
Ball_Radius = 7
WIN = pg.display.set_mode((WIDTH,HEIGHT))
pg.display.set_caption("Pong")
FPS = 60
SCORE_FONT = pg.font.SysFont("comicsans", 50)
Winning_Score = 3



WHITE = (255,255,255)
BLACK = (0,0,0)



def draw(win, paddles, ball, left_score, right_score):
    win.fill(BLACK)

    left_score_text = SCORE_FONT.render(f"{left_score}", 1, WHITE)
    right_score_text = SCORE_FONT.render(f"{right_score}", 1, WHITE)
    win.blit(left_score_text, (WIDTH//4 - left_score_text.get_width()//2, 20))
    win.blit(right_score_text, (WIDTH * (3/4) -
                                right_score_text.get_width()//2, 20))
    

    
    for i in range(10, HEIGHT, HEIGHT//20 ):
        if i%2  == 1:
            continue
        pg.draw.rect(win, WHITE, (WIDTH//2 - 5, i, 10, HEIGHT//20)) 

    for paddle in paddles:
        paddle.draw(win)
    
    ball.draw(win)

    pg.display.update()

class Ball:
    Max_Vel = 10
    Color = WHITE
    def __init__ (self, x,y, radius):
        self.x = self.original_x =  x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel =  self.Max_Vel
        self.y_vel = 0

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

class Paddle:
    Color = WHITE
    Vel = 4

    def __init__(self,x,y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y= y
        self.width= width
        self.height = height
        
    
    def draw(self,win):
        pg.draw.rect(win,self.Color, (self.x, self.y, self.width, self.height ))

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y



    def move(self, up = True):
        if up:
            self.y -= self.Vel
        else:
            self.y += self.Vel

def handle_paddle_movement(keys, left_paddle, right_paddle):
    if keys[pg.K_w] and left_paddle.y - left_paddle.Vel >= 0:
        left_paddle.move(up = True)
    if keys[pg.K_s] and left_paddle.y + left_paddle.Vel + left_paddle.height <= HEIGHT:
        left_paddle.move(up = False)
    if keys[pg.K_UP]  and right_paddle.y - right_paddle.Vel >= 0:
        right_paddle.move(up = True)
    if keys[pg.K_DOWN] and right_paddle.y + right_paddle.Vel + right_paddle.height <= HEIGHT:
        right_paddle.move(up = False)


def handle_collision(ball, left_paddle, right_paddle):
    if ball.y+ ball.radius >= HEIGHT:
        ball.y_vel *= -1
    elif ball.y - ball.radius <= 0:
        ball.y_vel *= -1

    if ball.x_vel < 0:
        if ball.y >= left_paddle.y and ball.y <= left_paddle.y + left_paddle.height:
            if ball.x - ball.radius <= left_paddle.x + left_paddle.width:
                ball.x_vel *= -1
                middle_y = left_paddle.y + left_paddle.height /2
                difference_in_y = middle_y - ball.y
                reduction_factor= (left_paddle.height /2) / ball.Max_Vel
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1* y_vel

    else:
        if ball.y >= right_paddle.y and ball.y <= right_paddle.y + right_paddle.height:
            if ball.x + ball.radius >= right_paddle.x:
                ball.x_vel *= -1
                middle_y = right_paddle.y + right_paddle.height /2
                difference_in_y = middle_y - ball.y
                reduction_factor= (right_paddle.height /2) / ball.Max_Vel
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1* y_vel




def main():
    clock = pg.time.Clock()
    run = True
    left_paddle = Paddle(10,HEIGHT//2 - Paddle_HEIGHT//2, Paddle_WIDTH, Paddle_HEIGHT )
    right_paddle = Paddle(WIDTH - 10 - Paddle_WIDTH ,HEIGHT//2 - Paddle_HEIGHT//2, Paddle_WIDTH, Paddle_HEIGHT )
    ball = Ball(WIDTH//2 , HEIGHT//2, Ball_Radius  )
    left_score = 0
    right_score = 0

    while run:
        clock.tick(FPS)
        draw(WIN, [left_paddle, right_paddle], ball, left_score, right_score)


        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                break
        
        keys = pg.key.get_pressed()
        
        handle_paddle_movement(keys,left_paddle, right_paddle )
        handle_collision(ball, left_paddle, right_paddle)
        if ball.x < 0: 
            right_score += 1
            pg.time.delay(500)
            ball.reset()
            left_paddle.reset()
            right_paddle.reset()
        elif ball.x > WIDTH:
            left_score += 1
            pg.time.delay(500)
            ball.reset() 
            left_paddle.reset()
            right_paddle.reset()

        won = False
        if left_score>= Winning_Score or  right_score>= Winning_Score:
            won = True
            win_text = "Game Over"
        
        if won:
            ball.reset()
            left_paddle.reset()
            right_paddle.reset()
            text = SCORE_FONT.render(win_text,1,WHITE)
            WIN.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height() //2 ))
            pg.display.update()
            pg.time.delay(5000)
            left_score = 0
            right_score = 0
            

        
            


        ball.move()


        
    pg.QUIT

if __name__ == '__main__':
    main()
