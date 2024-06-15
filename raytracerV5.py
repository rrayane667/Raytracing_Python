from util import *
import pygame

class raytracer:
    def __init__(self, pixels, resolution, camera, ray_depth) -> None:
        self.pixels = pixels
        self.hierarchy = []
        self.selection = []
        self.res = resolution
        self.WIDTH = len(pixels[0])
        self.HEIGHT = len(pixels)
        self.camera = [0,0,-500]
        self.depth = ray_depth
        
    def add_object(self, obj):
        self.hierarchy.append(obj)
        self.selection = [obj]
    
    def rayshooter(self, dir, origine):
        tMin = 10000
        vert = 0
        index = 0
        for object_ in self.hierarchy :
            if object_.isInside(origine, dir):
                vert_pos, t = object_.getCoord(dir, origine)
                if tMin >t:
                    vert = vert_pos
                    tMin = t
            index +=1
        return vert, index
    
    def material(self, vertex, index, incoming_ray):
        if self.hierarchy[index].isEmissive : return [255*self.hierarchy[index]]*3
        n = self.hierarchy[index].get_normal(vertex)
        viewVect = Normalize(diffVect(vertex, o))
        h = Normalize(addVect(viewVect, incoming_ray))
        color = multiVect(croi(BRDF(self.hierarchy[index].F0, n, h, viewVect, l, self.hierarchy[index].roughness, object_.color), [i*plight.int/(norm(diffVect(vert_pos, light_pos))**2 +0.00000001) for i in lightColor]), scalaire(n,l))
        return color
        
    def raytracer(self):
        
        def ray_recu(depth, origine, direction):
            vert, index = self.rayshooter(direction, origine)
            if depth == 0:
                return self.material(vert, index)
            
            return (0)*3 if not vert or not depth else ray_recu(depth - 1, vert, new_direction)
        
        for i in range(self.WIDTH//self.res):
            for j in range(self.HEIGHT//self.res):
                dir = Normalize([i*self.res - self.WIDTH/2, j*self.res - self.HEIGHT/2, -500-self.camera[2]])
        
    