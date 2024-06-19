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
                y = self.scale*cos(2*pi*i/self.subdivision) + self.position[1]
                z = self.scale*sin(2*pi*j/self.subdivision)*sin(2*pi*i/self.subdivision) + self.position[2]
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
                        pygame.draw.line(self.screen, [i*255 for i in obj.base_color], self.projection(coord[i]), self.projection(v))
            counter += 1



class raytracing:
    def __init__(self, pixels, resolution, camera, ray_depth, echantillon, width, height, dis_pix) -> None:
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
        #distance entre 2 pixel consecutif
        self.dis_pix = dis_pix

    def updateScene(self):
        self.lights = [object_ for object_ in self.hierarchy if object_.isEmissive == True]


    #adds object to the scene    
    def add_object(self, obj):
        self.hierarchy.append(obj)
        self.selection = [-1]

    def updateSelectionAttribute(self, position):
        self.hierarchy[self.selection].position = position
    
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
        if self.hierarchy[index].isEmissive : return [self.hierarchy[index].emissionIntensity*i for i in self.hierarchy[index].base_color]

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
            
            zeta = rd.uniform(0, 2*pi); r = sqrt(rd.uniform(0, 1))
            
            if n[0] != 0 and n[2] != 0 :
                #phi = arctan(n[1], n[2], n[0], n[1]); 
                psi = arctan(n[1], n[0],n[0])
                theta = atan(sqrt(n[0]**2 + n[1]**2)/n[2]) #arctan(sqrt(n[0]**2 + n[1]**2), n[2], n[2])
                #new_direction = (r*cos(zeta)*cos(theta)*cos(psi)-r*sin(zeta)*cos(theta)*sin(psi) - sqrt(1-r**2)*sin(theta), r*cos(phi)*sin(psi)*cos(zeta)+r*sin(zeta)*(cos(phi)*cos(psi)-sin(phi)*sin(psi)*sin(phi))+sqrt(1-r**2)*cos(theta)*sin(phi), r*cos(zeta)*(cos(phi)*cos(psi)*sin(theta) - sin(phi)*sin(psi))-r*sin(zeta)*(sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi)) + sqrt(1-r**2)*cos(theta)*cos(phi))
                new_direction = (r*cos(zeta)*cos(theta)*cos(psi) - r*sin(zeta)*cos(theta)*sin(psi)-sin(theta)*sqrt(1-r**2), r*cos(zeta)*sin(psi)+r*sin(zeta)+r*sin(zeta)*cos(psi),r*cos(zeta)*cos(psi)*sin(theta)-r*sin(psi)*sin(theta)*sin(zeta)+cos(theta)*sqrt(1-r**2))
            else: new_direction = (r*cos(zeta), r*sin(zeta), sqrt(1-r**2))
            
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
                dir = Normalize([i*self.res*self.dis_pix - self.WIDTH*self.dis_pix/2, j*self.res*self.dis_pix - self.HEIGHT*self.dis_pix/2, -500-self.camera[2]])
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
vert = (0,1,0)
bleu_ciel = (0,1,1)
WIDTH, HEIGHT = 500, 500
running = True
isWireframe = True
camera = [0, 0, -1300]

sphere1 = sphere([145, 0, -300], white, False, 0, [0.9]*3, 0.1, 80, 10)
sphere2 = sphere([-145, 0, 0], indigo, False, 0, [0.1]*3, 0.9, 120, 10)
sphere3 = sphere((0, 10000.0, 0.0), white, False, 1, [0.5]*3, 0.5, 9800, 50)
sphere4 = sphere((10000, 0, 0.0), bleu_ciel, False, 1, [0.7]*3, 0.2, 9780, 50)
sphere5 = sphere((-10000, 0, 0.0), red, False, 1, [0.2]*3, 0.5, 9740, 50)
sphere6 = sphere((0, -10000.0, 0.0), white, False, 0, [0.5]*3, 0.9, 9800, 50)
sphere6 = sphere((0, 0, 10000.0), white, True, 3, [0.5]*3, 0.9, 9850, 50)
sphere7 = sphere([20, 0, 50], white, True, 1, [0.9]*3, 0.1, 20, 4)
sphere8 = sphere((0, 0, -12400.0), white, True, 0, [0.8]*3, 0.2, 9850, 50)

res = 4
screen = pygame.display.set_mode((WIDTH, HEIGHT))
piksels = [[(0,0,0) for _ in range(WIDTH)] for _ in range(HEIGHT)]
dis_pix = 1

raytracer = raytracing(piksels, res, camera, 0, 10, WIDTH, HEIGHT, dis_pix)
wireframe_display = wireframe(abs(500 - camera[2]), screen, WIDTH, HEIGHT, camera)

raytracer.add_object(sphere1) 
raytracer.add_object(sphere2) 
raytracer.add_object(sphere3) 
raytracer.add_object(sphere4) 
raytracer.add_object(sphere5) 
raytracer.add_object(sphere6)
#raytracer.add_object(sphere7)

wireframe_display.add_object(sphere1) 
wireframe_display.add_object(sphere2) 
wireframe_display.add_object(sphere3) 
wireframe_display.add_object(sphere4) 
wireframe_display.add_object(sphere5) 
wireframe_display.add_object(sphere6)



while running :

    if isWireframe:
        wireframe_display.display()
    else:
        piksels = np.array(raytracer.raytracer())

        pygame.surfarray.blit_array(screen, piksels)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN :
            if event.button==1:
                isWireframe = False

    pygame.display.update()
pygame.quit()