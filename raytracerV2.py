import pygame
from math import sin ,cos
from util import *
from random import uniform

running = True

white = (255, 255, 255)
black = (0, 0, 0)
indigo = (75,0,130)
red = (255, 0, 0)
WIDTH, HEIGHT = 300, 300

screen = pygame.display.set_mode((WIDTH, HEIGHT))

c = [0, 0, -1300]

class sphere:
    def __init__(self, center, radius, color, F0, roughness) -> None:
        self.center = center
        self.radius = radius
        self.color = color
        self.step = None
        self.F0 = F0
        self.roughness = roughness
        
    def isInside(self, o, d):
        delta = sum([2*(o[i] - self.center[i])*d[i] for i in range(3)])**2 - 4*sum([i**2 for i in d])*(sum([(o[i] - self.center[i])**2 for i in range(3)]) - self.radius**2)
        if delta >=0: self.step = (-sum([2*(o[i] - self.center[i])*d[i] for i in range(3)]) - sqrt(delta))/(2*sum([i**2 for i in d])); return True
        return False
    
    def getCoord(self, dir, c):
        return [c[0] + self.step*dir[0], c[1] + self.step*dir[1], c[2] + self.step*dir[2]]#[c[i] + self.step*d[i] for i in range(3)]
    
    def getNormal(self, v):
        return Normalize(diffVect( self.center, v))
    
class light:
    def __init__(self, intensite, pos):
        self.int = intensite
        self.pos = pos

sphere1 = sphere([145, 0, 0], 80, indigo, [0.9]*3, 0.01)
sphere2 = sphere([-145, 0, -0], 50, white, [0.9]*3, 0.01)
plight = light(12000, multiVect(Normalize([0, 0, -1]), 200))
        

hierarchy = [sphere1, sphere2]


def rayShooter(o,d, light_pos, object_, lightColor):
    color = [0]*3
    if object_.isInside(o, d):
        vert_pos = object_.getCoord(d, o)
        n = object_.getNormal(vert_pos)
        l = Normalize(diffVect(vert_pos, light_pos))
        viewVect = Normalize(diffVect(vert_pos, o))
        h = Normalize(addVect(viewVect, l))
        color = multiVect(croi(BRDF(object_.F0, n, h, viewVect, l, object_.roughness, object_.color), [i*plight.int/(norm(diffVect(vert_pos, light_pos))**2 +0.00000001) for i in lightColor]), scalaire(n,l))
        return color, vert_pos
    return color, [12, -3, 1]

def rqytrqcing(camera_pos, d, light_pos, sample):
    object1 = hierarchy[0]
    object2 = hierarchy[1]
    if object1.isInside(camera_pos, d):
        vertexHit = object1.getCoord(d, camera_pos)
        n = object1.getNormal(vertexHit)
        
        eclairageDirecte = rayShooter(camera_pos, d, light_pos, object1, [255]*3)[0]
        finalColor = eclairageDirecte

        
        for _ in range(sample):
            theta = uniform(0, 2*pi); phi = uniform(0, pi/2) # coordonnée sphérique vecteur unitaire
            r = [sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)]
            new_d = [n[i] - r[i] for i in range(3)]
            couleurReflechi, sourceRef = rayShooter(vertexHit, new_d, plight.pos, object2, [255]*3)

            couleurObjet = rayShooter(camera_pos, d, sourceRef, object1, couleurReflechi)[0]
            finalColor = addVect(finalColor, couleurObjet)  
        return multiVect(finalColor, 2*pi/(sample+1)) 
    
    if object2.isInside(camera_pos, d):
        '''vertexHit = object2.getCoord(d, camera_pos)
        n = object2.getNormal(vertexHit)
        l = Normalize(diffVect(vertexHit, light_pos))
        viewVect = Normalize(diffVect(vertexHit, camera_pos))
        h = Normalize(addVect(viewVect, l))
        eclairageDirecte = rayShooter(camera_pos, d, light_pos, object2)
        finalColor = eclairageDirecte
        for _ in range(depth):
            theta = uniform(0, 2*pi); phi = uniform(0, pi/2) # coordonnée sphérique vecteur unitaire
            r = [sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)]
            new_d = [n[i] - r[i] for i in range(3)]
            couleurReflechi = rayShooter(vertexHit, new_d, plight.pos, object1)
            couleurObjet = multiVect(croi(BRDF(object2.F0, n, h, viewVect, l, object2.roughness, object2.color), [i*plight.int/(norm(diffVect(vertexHit, light_pos))**2 +0.00000001) for i in couleurReflechi]), scalaire(n,l))
            finalColor = addVect(finalColor, couleurObjet)  
        return multiVect(finalColor, 2*pi/(depth+1))'''
        return rayShooter(camera_pos, d, light_pos, object2, [255]*3)[0]
    return [0]*3
        
    
    
for i in range(WIDTH):
    for j in range(HEIGHT):
        dir = Normalize([i - WIDTH/2, j - HEIGHT/2, -500-c[2]])
        color = rqytrqcing(c, dir, plight.pos, 4)
        
        if norm(color) : screen.set_at((i,j), [max(0, min(i,255)) for i in color])
        
    



while running :

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
pygame.quit()