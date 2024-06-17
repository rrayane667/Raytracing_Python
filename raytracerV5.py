from util import *
import pygame
import random as rd
from math import cos, sin, sqrt, acos
import numpy as np
import sys

sys.setrecursionlimit(2056)

class objects_:
    def __init__(self,position, base_color, isEmissive, emissionIntensity, F0, roughness) -> None:
        self.position = position
        self.base_color = base_color
        self.isEmissive = isEmissive
        self.emissionIntensity = emissionIntensity
        self.F0 = F0
        self.roughness = roughness
        self.step = 0

class sphere(objects_):
    def __init__(self, position, base_color, isEmissive, emissionIntensity, F0, roughness, radius) -> None:
        self.radius = radius
        super().__init__(position, base_color, isEmissive, emissionIntensity, F0, roughness)

    def isInside(self, o, d):
        delta = sum([2*(o[i] - self.position[i])*d[i] for i in range(3)])**2 - 4*sum([i**2 for i in d])*(sum([(o[i] - self.position[i])**2 for i in range(3)]) - self.radius**2)
        if delta >=0: self.step = (-sum([2*(o[i] - self.position[i])*d[i] for i in range(3)]) - sqrt(delta))/(2*sum([i**2 for i in d])); return True
        return False

    def getNormal(self, vertice):
        return Normalize(diffVect( vertice, self.position))

    def getCoord(self, dir ,c):
        return [c[0] + self.step*dir[0], c[1] + self.step*dir[1], c[2] + self.step*dir[2]], self.step


class raytracing:
    def __init__(self, pixels, resolution, camera, ray_depth, echantillon, width, height) -> None:
        #2D array containing the pixels
        self.pixels = pixels
        #array containing objects of the scene
        self.hierarchy = []
        #array containing all the objects with emissive material
        self.lights = []
        #array containing the object currently selected
        self.selection = []
        #integer defining the resolution of the final image (res = n means that only 1/n of the original pixels are computed, the rest is filled with the walus of the neighboring pixels)
        self.res = resolution
        #Width of the image
        self.WIDTH = width
        #height of the image
        self.HEIGHT = height
        #position of the camera
        self.camera = camera
        #number of reflection for a given ray
        self.depth = ray_depth
        #number of rays generated for each reflection
        self.echantillon = echantillon

    def updateScene(self):
        self.lights = [object_ for object_ in self.hierarchy if object_.isEmissive == True]


    #adds object to the scene    
    def add_object(self, obj):
        self.hierarchy.append(obj)
        self.selection = [obj]
    
    #send ray and calculate intersections
    def rayshooter(self, dir, origine):
        
        tMin = 10000
        vert = 0
        i = 0
        index = 0
        for object_ in self.hierarchy :
            if object_.isInside(origine, dir):
                vert_pos, t = object_.getCoord(dir, origine)
                if tMin >t:
                    vert = vert_pos
                    tMin = t
                    index = i
            i += 1
        return vert, index
    
    def ray_recu(self, depth, origine, direction, vert, index):
        if self.hierarchy[index].isEmissive : return [self.hierarchy[index].emissionIntensity]*3

            #gets the normal vector at the position of the vertex
        n = self.hierarchy[index].getNormal(vert)
        OBJcolor = (0,0,0)
        
        if depth <= 0  and vert:
            for light in self.lights:
                l = Normalize(diffVect(vert, light.position))
                OBJcolor = addVect(OBJcolor,self.material(vert, index, l, origine, [light.emissionIntensity for i in light.base_color], n))
            OBJcolor = multiVect(OBJcolor, 2*pi/len(self.lights))
            return OBJcolor
        if depth <= 0:
            return OBJcolor
 

        for _ in range(self.echantillon):
            current_color = (0,0,0)
            #choosing r and theta (polar coordinates) using uniformly distributed values doesnt produce uniformly distributed vectors in space, needs rework
            theta = rd.uniform(0, 2*pi); phi = direction[2]; z = rd.uniform(phi, 1) # coordonnée sphérique vecteur unitaire
            new_direction = Normalize(multiVect(addVect([sqrt(1-z**2)*cos(theta), sqrt(1-z**2)*sin(theta), z],vert), 1))
            new_vert, new_index = self.rayshooter(new_direction, vert)
            if new_vert: 
                new_l = Normalize(diffVect(vert, new_vert))
                current_color = self.material(vert, index, new_l, origine, self.ray_recu(depth-1, vert, new_direction, new_vert, new_index),n)
                OBJcolor = addVect(OBJcolor, current_color)
        OBJcolor = multiVect(OBJcolor, 2*pi/self.echantillon)
                
        return OBJcolor

    #calculate the material using the object attributes
    def material(self, vertex, index, l, camera, next_color, n):
        #vector from camera to vertex
        viewVect = Normalize(diffVect(vertex, camera))
        h = Normalize(addVect(viewVect, l))
        color = multiVect(croi(BRDF(self.hierarchy[index].F0, n, h, viewVect, l, self.hierarchy[index].roughness, self.hierarchy[index].base_color), next_color), scalaire(n,l))
        return color
    
    def raytracer(self):
        self.updateScene()
        
        for i in range(self.WIDTH//self.res):
            for j in range(self.HEIGHT//self.res):
                dir = Normalize([i*self.res - self.WIDTH/2, j*self.res - self.HEIGHT/2, -500-self.camera[2]])
                vertice, indice = self.rayshooter(dir, self.camera)
                color = (0,0,0)
                if vertice :
                    color = self.ray_recu(self.depth, self.camera, dir, vertice, indice)

                if norm(color) :
                    for k in range(self.res):
                        for l in range(self.res):
                            self.pixels[i*self.res + k][j*self.res + l] = [min(col,1)*255 for col in color]
        return self.pixels
        
white = (1, 1, 1)
black = (0, 0, 0)
indigo = (75/255,0,130/255)
red = (1, 0, 0)
jaune = (1, 200/255, 0)
WIDTH, HEIGHT = 500, 500
running = True
camera = [0, 0, -1300]

sphere1 = sphere([145, 0, -300], white, True, 1, [0.9]*3, 0.1, 80)
sphere2 = sphere([-145, 0, 0], indigo, False, 1, [0.7]*3, 0.4, 120)
sphere3 = sphere((0, 10000.0, 0.0), jaune, False, 1, [0.5]*3, 0.5, 9800)
sphere4 = sphere((10000, 0, 0.0), red, False, 1, [0.5]*3, 0.5, 9780)
sphere5 = sphere((-10000, 0, 0.0), red, False, 1, [0.5]*3, 0.5, 9740)
sphere6 = sphere((0, -10000.0, 0.0), white, False, 1, [0.5]*3, 0.5, 9850)

res = 10
screen = pygame.display.set_mode((WIDTH, HEIGHT))
piksels = [[(0,0,0) for _ in range(WIDTH)] for _ in range(HEIGHT)]

raytracer = raytracing(piksels, res, camera, 1, 100, WIDTH, HEIGHT)

raytracer.add_object(sphere1)
raytracer.add_object(sphere2)
raytracer.add_object(sphere3)
raytracer.add_object(sphere4)
raytracer.add_object(sphere5)
raytracer.add_object(sphere6)

piksels = np.array(raytracer.raytracer())

pygame.surfarray.blit_array(screen, piksels)

while running :

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
pygame.quit()