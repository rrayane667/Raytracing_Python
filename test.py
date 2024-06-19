from util import *
import pygame
import random as rd
from math import cos, sin, sqrt, acos, atan
import numpy as np
import sys

sys.setrecursionlimit(2056)

class objects_:
    def __init__(self,position, base_color, isEmissive, emissionIntensity, F0, roughness, scale, name, subdivision) -> None:
        self.position = position
        self.base_color = base_color
        self.isEmissive = isEmissive
        self.emissionIntensity = emissionIntensity
        self.F0 = F0
        self.roughness = roughness
        self.step = 0
        self.scale = scale
        self.subdivision = subdivision

class sphere(objects_):
    def __init__(self, position, base_color, isEmissive, emissionIntensity, F0, roughness, scale, subdivision) -> None:
        self.coord = []
        super().__init__(position, base_color, isEmissive, emissionIntensity, F0, roughness, scale, "Sphere", subdivision)

    def isInside(self, o, d):
        delta = sum([2*(o[i] - self.position[i])*d[i] for i in range(3)])**2 - 4*sum([i**2 for i in d])*(sum([(o[i] - self.position[i])**2 for i in range(3)]) - self.scale**2)
        if delta >=0: self.step = (-sum([2*(o[i] - self.position[i])*d[i] for i in range(3)]) - sqrt(delta))/(2*sum([i**2 for i in d])); return True
        return False

    def getNormal(self, vertice):
        return Normalize(diffVect( self.position, vertice))

    def getCoord(self, dir ,c):
        return [c[0] + self.step*dir[0], c[1] + self.step*dir[1], c[2] + self.step*dir[2]], self.step
    
    def wireframe_coord(self):
        for i in range(self.subdivision):
            for j in range(self.subdivision):
                x = self.scale*cos(2*pi*j/self.subdivision)*sin(2*pi*i/self.subdivision) + self.position[0]
                y =  self.scale*sin(2*pi*j/self.subdivision)*sin(2*pi*i/self.subdivision) + self.position[1]
                z = self.scale*cos(2*pi*i/self.subdivision) + self.position[2]
                self.coord.append([x,y,z])
    def edges(self, v_index):
        return [self.coord[v_index - self.subdivision], self.coord[v_index - 1]]

class wireframe:
    def __init__(self, camera_dist, screen, HEIGHT, WIDTH, camera) -> None:
        self.camera_dist = camera_dist
        self.screen = screen
        self. hierarchy = []
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.camera = camera

    def rayshooter(self, vertice, index):
        self.hierarchy[index].isInside(self.camera, Normalize(diffVect(self.camera, vertice)))
        object_step = self.hierarchy[index].step
        for i, obj in enumerate(self.hierarchy):
            if obj.isInside(self.camera, Normalize(diffVect(self.camera, vertice))) and i != index and obj.step < object_step:
                return True
        return False

    def add_object(self, obj):
        self.hierarchy.append(obj)

    def igrec(self, z, y):
        return(self.camera_dist*y/(z+self.camera_dist)+self.HEIGHT/2)

    def ix(self, z, x):
        return(self.camera_dist*x/(z+self.camera_dist)+self.WIDTH/2)
    
    def projection(self, v):
        return (self.ix(v[2], v[0]), self.igrec(v[2], v[1]))
    
    def display(self):
        counter = 0
        for obj in self.hierarchy :
            obj.wireframe_coord()
            coord = obj.coord
            for i in range(0,len(coord)):
                normal = obj.getNormal(coord[i])
                view_vector = Normalize(diffVect(coord[i], self.camera))
                if i%obj.subdivision and scalaire(normal, view_vector) and not self.rayshooter(coord[i], counter):
                    edges = obj.edges(i)
                    for v in edges:
                        pygame.draw.line(self.screen, (255,255,255), self.projection(coord[i]), self.projection(v))
            counter += 1

running = True
screen = pygame.display.set_mode((300, 300))
sphere1 = sphere((100,0,0), (255,255,255), False, 0, 0,0,50,20)
sphere2 = sphere((-100,0,200), (255,255,255), False, 0, 0,0,50,30)
sphere3 = sphere((-100,0,0), (255,255,255), False, 0, 0,0,50,10)
affichage = wireframe(1300, screen, 300, 300, (0,0, -1300))

affichage.add_object(sphere1)
affichage.add_object(sphere2)
affichage.add_object(sphere3)


affichage.display()

while running :

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
pygame.quit()