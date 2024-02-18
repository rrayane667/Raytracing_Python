import pygame


from math import sqrt

white = (255, 255, 255)
black = (0, 0, 0)
indigo = (75,0,130)
c = [0, 0, -1300]

class light:
    def __init__(self, lightType, lightIntensity, lightPosition, innerRadiusAngle, outerRadiusAngle, direction):
        self.lightType = lightType # lightType = 0 -> directional light, 1 ->spot light, 2 -> point light
        self.lightIntensity = lightIntensity
        self.lightPosition = lightPosition
        self.innerRadiusAngle = innerRadiusAngle # only works for spot light and point light, for spot light it defines the angle between the direction and the edge of the cone with max intensity de meme for outerRadiusAngle, for the point light it defines the radius of the sphere where max intensity
        self.outerRadiusAngle = outerRadiusAngle # only works for spot and point light, for point light light outerRadius = innerRadius * outerRadiusAngle
        self.direction = direction # only works for spot and directional light

spotLight = light(1, 1, [0, 0, -400], 0.95, 0.6, [0, 0, 1])
pointLight = light(2, 0, [0, 0, -400], 320, 1.9, [0, 0, 1])

running = True
WIDTH, HEIGHT = 400, 400




def Normalize(u):
    r = sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    if not r: return u
    return [u[0]/r, u[1]/r, u[2]/r]

def scalaire(v, u): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def diffVect(v, u):
    return [u[0]-v[0], u[1]-v[1], u[2]-v[2]]

def addVect(u, v):
    return [u[0]+v[0], u[1]+v[1], u[2]+v[2]]

def multiVect(u, x):
    return [u[0]*x, u[1]*x, u[2]*x]

def linearInterpolation( a, b, t):
    return addVect(multiVect(a,(1-t)) ,multiVect(b,t))

def isInside(c, d, r):
    delta = sum([2*c[i]*d[i] for i in range(3)])**2 - 4*sum([i**2 for i in d])*(sum([i**2 for i in c]) - r**2)
    if delta >=0: return [True, (-sum([2*c[i]*d[i] for i in range(3)]) - sqrt(delta))/(2*sum([i**2 for i in d]))]
    return [False, None]

screen = pygame.display.set_mode((WIDTH, HEIGHT))


for i in range(WIDTH):
    for j in range(HEIGHT):
        dir = Normalize([i - WIDTH/2, j - HEIGHT/2, -500+1300])
        r=300
        s = isInside([0, 0, -1300], dir, r)
        if s[0] :
            v = [c[0] + s[1]*dir[0], c[1] + s[1]*dir[1], -s[1]*dir[2]]
            n = Normalize(v)
            v = multiVect(n, r)
            l = Normalize(diffVect(v,multiVect(Normalize([1/12, 0,-1]), r + 20) ))
            cam = Normalize(diffVect(v, c))
            h = Normalize(addVect(l, cam))
            spe = scalaire(n, h)**100
            vis = max(0,scalaire(n, l))
            color = linearInterpolation(indigo, white, spe)

            screen.set_at((i, j), multiVect(color, vis))








while running :

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
pygame.quit()