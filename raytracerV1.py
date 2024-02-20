import pygame


from math import sqrt, pi, ceil

white = (255, 255, 255)
black = (0, 0, 0)
indigo = (75,0,130)
red = (255, 0, 0)


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
WIDTH, HEIGHT = 600, 600




def Normalize(u):
    r = sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    if not r: return u
    return [u[0]/r, u[1]/r, u[2]/r]

def scalaire(v, u): return max(0,u[0]*v[0] + u[1]*v[1] + u[2]*v[2])

def diffVect(v, u):
    return [u[0]-v[0], u[1]-v[1], u[2]-v[2]]

def addVect(u, v):
    return [u[0]+v[0], u[1]+v[1], u[2]+v[2]]

def multiVect(u, x):
    return [u[0]*x, u[1]*x, u[2]*x]

def croi(X, Y):
    return [X[i]*Y[i] for i in range(len(X))]

def linearInterpolation( a, b, t):
    return addVect(multiVect(a,(1-t)) ,multiVect(b,t))

def norm(u):
    return sqrt(u[0]**2 + u[1]**2 + u[2]**2)

def sym(X, N):
    scal = scalaire(X, N)
    return [(X[i] - 2*scal*N[i]) for i in range(len(X)) ]

def isInsideSphere(c, d, r):
    delta = sum([2*c[i]*d[i] for i in range(3)])**2 - 4*sum([i**2 for i in d])*(sum([i**2 for i in c]) - r**2)
    if delta >=0: return [True, (-sum([2*c[i]*d[i] for i in range(3)]) - sqrt(delta))/(2*sum([i**2 for i in d]))]
    return [False, None]

def isInsidePlane(c, d, y):
    if d[1]<=0: return [False, None]
    #intersection vecteur incident et ray
    x0 = c[0] + (y-c[1])*d[0]/(d[1])
    y0 = y
    z0 = c[2] + (y-c[1])*d[2]/(d[1])
    return [True, [x0, y0, z0]]



#Rendering Functions :
def D(roughness, N, H): #scalaire
    return (roughness**2)/(pi*((scalaire(N, H)**2)*(roughness**2 - 1) +1)**2 + 0.00001)

def G1(N, X, roughness): #scalaire
    k = (roughness**2)/2
    return scalaire(N,X)/(scalaire(N, X)*(1-k) + k)

def G(N, V, L, roughness): #scalaire
    return G1(N, V, roughness)*G1(N, L, roughness)

def F(F0, N, H): #Vecteur
    return F0 + multiVect(diffVect( F0, [1, 1, 1]), (1-scalaire(N, H))**5)

def Flamb(c): #vecteur
    return multiVect(c, 1/pi)

def Fspe( N, H, V, L, roughness): #scalaire
    return D(roughness, N, H)*G(N, V, L, roughness)/(4*scalaire(V, N)*scalaire(N, L) + 0.00001)

def BRDF(F0, N, H, V, L, roughness, c):
    Ks = F(F0, N, H)
    Kd = diffVect(Ks, [1, 1, 1])
    fd = croi(Kd, Flamb(c))
    fs = multiVect(Ks, Fspe(N, H, V, L, roughness))
    return addVect(fd, fs)



screen = pygame.display.set_mode((WIDTH, HEIGHT))

c = [0, 0, -1300]
F0 = [0.2]*3
r=300
roughness = 0.1
intensité_lum = 12000
light_pos = multiVect(Normalize([1, -1, -1]), 900)
hauteur =  r+20

F0plane = [0.2]*3
roughnessPlane = 0.9
for i in range(WIDTH):
    for j in range(HEIGHT):
        dir = Normalize([i - WIDTH/2, j - HEIGHT/2, -500-c[2]])
        
        # plane rendering
        planInterRay = isInsidePlane(c, dir,  r+20) # coord intersection entre plan et rayon incidant
        vert_pos = planInterRay[1]
        if planInterRay[0]:
            #shadows
            s = isInsideSphere(planInterRay[1], Normalize(diffVect(vert_pos, light_pos)), r)
            if not s[0]:

                

                colorTest = black
                pvDotpl = 0
                #calcul reflection
                s = isInsideSphere(planInterRay[1], [dir[0], -dir[1], dir[2]], r)
                if s[0]:
                    vert_pos_sphere = [c[0] + s[1]*dir[0], c[1] + s[1]*dir[1], c[2] + s[1]*dir[2]]
                    n = Normalize(vert_pos_sphere)
                    vert_pos = multiVect(n, r)
                    dist_light = norm(diffVect(vert_pos_sphere,light_pos ))
                    l = Normalize(diffVect(vert_pos_sphere,light_pos ))
                    viewVect = Normalize(diffVect(vert_pos_sphere, vert_pos))
                    h = Normalize(addVect(l, viewVect))
                    color = multiVect(croi(BRDF(F0, n, h, viewVect, l, roughness, indigo), [intensité_lum*255/(dist_light)**2]*3),scalaire(n,l))
                    colorTest = [min(255, color[i]) for i in range(3)]
                    
                    n = [0, -1, 0]
                    dist_light = norm(diffVect(vert_pos_sphere, vert_pos))
                    l = Normalize(diffVect(vert_pos, vert_pos_sphere))
                    viewVect = Normalize(diffVect(vert_pos, c))
                    h = Normalize(addVect(l, viewVect))
                    colorTest = multiVect(croi(BRDF(F0plane, n, h, viewVect, l, roughnessPlane, indigo), [i/(dist_light)**2 for i in colorTest]*3),scalaire(n,l))
                    
                #colorTest = indigo
                
                #vcDotvp = scalaire(Normalize(diffVect(vert_pos, c)),Normalize(diffVect( vert_pos,p)))
                #colorTest = multiVect(colorTest, pvDotpl)

                    #screen.set_at((i, j), colorTest)


                #calcul couleur eclairage direct
                
                n = [0, -1, 0]
                dist_light = norm(diffVect(vert_pos, light_pos))
                l = Normalize(diffVect(vert_pos,light_pos ))
                viewVect = Normalize(diffVect(vert_pos, c))
                h = Normalize(addVect(l, viewVect))
                color = multiVect(croi(BRDF(F0plane, n, h, viewVect, l, roughnessPlane, white), [intensité_lum*255/(dist_light)**2]*3),scalaire(n,l))
                color = [min(255, color[i]) for i in range(3)]

                
                
                screen.set_at((i, j), [min(255, i) for i in addVect(color, colorTest)])


        #sphere rendering
        s = isInsideSphere(c, dir, r)
        if s[0] :
            vert_pos = [c[0] + s[1]*dir[0], c[1] + s[1]*dir[1], c[2] + s[1]*dir[2]]
            n = Normalize(vert_pos)
            
            dist_light = norm(diffVect(vert_pos,light_pos ))
            l = Normalize(diffVect(vert_pos,light_pos ))
            viewVect = Normalize(diffVect(vert_pos, c))
            h = Normalize(addVect(l, viewVect))
            color = multiVect(croi(BRDF(F0, n, h, viewVect, l, roughness, indigo), [intensité_lum*255/(dist_light)**2]*3),scalaire(n,l))
            colorSphere = [min(255, color[i]) for i in range(3)]

            
            rayon_reflechi = Normalize(sym(dir, n))
            insidePlane = isInsidePlane(vert_pos, rayon_reflechi, hauteur)
            planeVert = insidePlane[1]
            t=0
            colorReflechi = [0]*3

            if insidePlane[0]:
                t=1/2
                #lighting of the plane
                dist_light = norm(diffVect(planeVert,light_pos ))
                l = Normalize(diffVect(planeVert,light_pos ))
                viewVect = Normalize(diffVect(planeVert, vert_pos))
                h = Normalize(addVect(l, viewVect))
                colorP = multiVect(croi(BRDF(F0, [0, -1, 0], h, viewVect, l, roughness, white), [intensité_lum*255/(dist_light)**2]*3),scalaire(n,l))
                colorplan = [min(255, color[i]) for i in range(3)]

                #reflection
                dist_light = norm(diffVect(vert_pos,planeVert ))
                l = multiVect(Normalize(diffVect(vert_pos,planeVert )), -1)
                viewVect = multiVect(Normalize(diffVect(vert_pos, c)), -1)
                h = Normalize(addVect(l, viewVect))
                color = multiVect(croi(BRDF(F0, n, h, viewVect, l, roughness, indigo), [intensité_lum*i/(dist_light)**2 for i in colorplan]),scalaire(n,l))#intensité_lum*i/(dist_light)**2
                colorReflechi = [min(255, color[i]) for i in range(3)]


            finalColor = addVect(colorSphere, colorReflechi)#linearInterpolation(colorSphere, colorReflechi, t)
            finalColor = [min(255, i) for i in finalColor]




            screen.set_at((i, j), finalColor)

        









while running :

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
pygame.quit()