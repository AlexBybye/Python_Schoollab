import turtle

screen = turtle.Screen()
screen.setup(width=600, height=600)     
t = turtle.Turtle()

def draw_square(size):
    for _ in range(4):
        t.forward(size)
        t.left(90)

square_size = 100
num_rotations = 24
rotation_angle = 360 / num_rotations

for _ in range(num_rotations):
    draw_square(square_size)
    t.left(rotation_angle) 

screen.mainloop()