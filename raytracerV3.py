import pygame
from math import sin ,cos
from util import *
from random import uniform

running = True

white = (255, 255, 255)
black = (0, 0, 0)
indigo = (75,0,130)
red = (255, 0, 0)
jaune = (255, 200, 0)
WIDTH, HEIGHT = 300, 300

res = 3

screen = pygame.display.set_mode((WIDTH, HEIGHT))

c = [0, 0, -1300]
class quad:
    def __init__(self, hauteur, longueur, color, F0, roughness, isEmissive, emissionIntensity) -> None:
        self.hauteur = hauteur
        self.longueur = longueur
        self.color = color
        self.F0 = F0
        self.roughness = roughness
        self.isEmissive = isEmissive
        self.emissionIntensity = emissionIntensity
        self.Coord = None
        
        
    def normInfiniXZ(self, v):
        return max(abs(v[0]), abs(v[2]))
    
    def getNormal(self, v):
        return [0, -1, 0]
    
    def getCoord(self, d, o):
        return self.Coord
    
    def isInside(self, o, d):
        if d[1]*self.hauteur<=0: return False
        x0 = o[0] + (self.hauteur-o[1])*d[0]/(d[1])
        y0 = self.hauteur
        z0 = o[2] + (self.hauteur-o[1])*d[2]/(d[1])
        if self.normInfiniXZ([x0, y0, z0])>self.longueur: return False
        self.Coord = [x0, y0, z0]
        return True
        

class sphere:
    def __init__(self, center, radius, color, F0, roughness, isEmissive , emissionIntensity) -> None:
        self.center = center
        self.radius = radius
        self.color = color
        self.step = None
        self.F0 = F0
        self.roughness = roughness
        self.isEmissive = isEmissive
        self.emissionIntensity = emissionIntensity
        
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

plan = quad(85, 4000, jaune, [0.9]*3, 0.1, False, 1)
plan2 = quad(-200, 200, white, [0.4]*3, 0.1, True, 50)
sphere1 = sphere([145, 0, 0], 80, indigo, [0.999]*3, 0.1, False, 0)
sphere2 = sphere([-145, 0, -0], 50, red, [0.99]*3, 0.1, False, 0)
sphere3 = sphere([0, -80, -500], 68, white, [0.9]*3, 0.01, True, 50)
plight = light(12000, multiVect(Normalize([0, 0, -1]), 200))
        

hierarchy = [sphere1, sphere2, plan, plan2]


def rayShooter(o,d, light_pos, object_, lightColor):
    color = [0]*3
    
    if object_.isInside(o, d):
        vert_pos = object_.getCoord(d, o)
        if object_.isEmissive : return [255*object_.emissionIntensity]*3, vert_pos
        n = object_.getNormal(vert_pos)
        l = Normalize(diffVect(vert_pos, light_pos))
        viewVect = Normalize(diffVect(vert_pos, o))
        h = Normalize(addVect(viewVect, l))
        color = multiVect(croi(BRDF(object_.F0, n, h, viewVect, l, object_.roughness, object_.color), [i*plight.int/(norm(diffVect(vert_pos, light_pos))**2 +0.00000001) for i in lightColor]), scalaire(n,l))
        return color, vert_pos
    return color, [12, -3, 1]

def rqytrqcing(camera_pos, d, sample):

    for i, object_ in enumerate(hierarchy):
        if object_.isInside(camera_pos, d):
            vertexHit = object_.getCoord(d, camera_pos)
            n = object_.getNormal(vertexHit)

            finalColor = [0]*3

            
            for _ in range(sample):
                theta = uniform(0, 2*pi); phi = uniform(0, pi/2) # coordonnée sphérique vecteur unitaire
                r = [sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)]
                new_d = [n[i] - r[i] for i in range(3)]
                sourceRef = [12, -3, 1]
                couleurReflechi = [0]*3
                for object2 in [hierarchy[k] for k in range(len(hierarchy)) if k!=i]:
                    if object2.isInside(vertexHit, new_d): couleurReflechi, sourceRef = rayShooter(vertexHit, new_d, plight.pos, object2, [255]*3); break

                couleurObjet = rayShooter(camera_pos, d, sourceRef, object_, couleurReflechi)[0]
                finalColor = addVect(finalColor, couleurObjet)  
            return multiVect(finalColor, 2*pi/(sample)) 
    return [0]*3
        
    
    
for i in range(WIDTH//res):
    for j in range(HEIGHT//res):
        dir = Normalize([i*res - WIDTH/2, j*res - HEIGHT/2, -500-c[2]])
        color = rqytrqcing(c, dir, 100)
        
        if norm(color) : 
            for k in range(res):
                for l in range(res):
                    screen.set_at((i*res + k,j*res + l), [max(0, min(i,255)) for i in color])
        
    



while running :

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
pygame.quit()