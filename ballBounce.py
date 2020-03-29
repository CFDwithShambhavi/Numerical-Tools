#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:20:18 2020

@author: snandan
"""

import turtle as tr
import random as ran

box2D = tr.Screen()
box2D.bgcolor('white')
box2D.title('Perfect Elastic Bouncing Ball')
box2D.tracer(0)

balls = []

for b in range(20):
    balls.append(tr.Turtle())
    
colors = ['red', 'blue', 'cyan', 'magenta', 'orange', 'purple', 'green']
shapes = ['circle', 'square', 'triangle']

for ball in balls:
    ball.shape(ran.choice(shapes))
    ball.color(ran.choice(colors))
    ball.penup()
    ball.speed(0)
    x = ran.randint(-300, 300)
    y = ran.randint(200, 400)
    ball.goto(x, y)
    ball.delY = 0
    ball.delx = ran.randint(-3, 3)
    ball.dtheta = ran.randint(-5, 5)

gravity = 0.2
while True:
    box2D.update()
    for ball in balls:
        ball.rt(ball.dtheta)
        ball.delY -= gravity
        ball.sety(ball.ycor() + ball.delY)
        
        ball.setx(ball.xcor() + ball.delx)
        
        if ball.xcor() > 325:
            ball.delx *= -1
            ball.dtheta *= -1
        
        if ball.xcor() < -325:
            ball.delx *= -1
            ball.dtheta *= -1
        
        if ball.ycor() < -300:
            ball.sety(-300)
            ball.delY *= -1
            ball.dtheta *= -1
            
        #if ball.ycor() > 300:
            #ball.delY *= -1

box2D.exitonclick()