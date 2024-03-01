import numpy as np
from numba import cuda, float32, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import pygame
from random import uniform


# Util functions adapted for CUDA
@cuda.jit(device=True)
def normalize(v):
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if norm > 0:
        inv_norm = 1.0 / norm
        return (v[0] * inv_norm, v[1] * inv_norm, v[2] * inv_norm)
    return v

@cuda.jit(device=True)
def scalaire(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

@cuda.jit(device=True)
def diffVect(v2, v1):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

@cuda.jit(device=True)
def addVect(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])

@cuda.jit(device=True)
def multiVect(u, x):
    return (u[0] * x, u[1] * x, u[2] * x)

@cuda.jit(device=True)
def croi(X, Y):
    return (X[0] * Y[0], X[1] * Y[1], X[2] * Y[2])

@cuda.jit(device=True)
def linearInterpolation( a, b, t):
    return addVect(multiVect(a,(1-t)) ,multiVect(b,t))

@cuda.jit(device=True)
def norm(u):
    return math.sqrt(u[0]**2 + u[1]**2 + u[2]**2)

@cuda.jit(device=True)
def sym(X, N):
    scal = scalaire(X, N)
    return ((X[0] - 2*scal*N[0]), (X[1] - 2*scal*N[1]), (X[2] - 2*scal*N[2]))

@cuda.jit(device=True)
def D(roughness, N, H): #scalaire
    return (roughness**2)/(math.pi*((scalaire(N, H)**2)*(roughness**2 - 1) +1)**2 + 0.00001)

@cuda.jit(device=True)
def G1(N, X, roughness): #scalaire
    k = (roughness*roughness)/2
    return scalaire(N,X)/(scalaire(N, X)*(1-k) + k)

@cuda.jit(device=True)
def G(N, V, L, roughness): #scalaire
    return G1(N, V, roughness)*G1(N, L, roughness)

@cuda.jit(device=True)
def F(F0, N, H):
    return addVect(F0, multiVect(diffVect( F0, (1, 1, 1)), (1-scalaire(N, H))**5))

@cuda.jit(device=True)
def Flamb(c): #vecteur
    return multiVect(c, 1/math.pi)

@cuda.jit(device=True)
def Fspe( N, H, V, L, roughness): #scalaire
    return D(roughness, N, H)*G(N, V, L, roughness)/(4*scalaire(V, N)*scalaire(N, L) + 0.00001)

@cuda.jit(device=True)
def BRDF(F0, N, H, V, L, roughness, c):
    Ks = F(F0, N, H)
    Kd = diffVect(Ks, (1, 1, 1))
    fd = croi(Kd, Flamb(c))
    fs = multiVect(Ks, Fspe(N, H, V, L, roughness))
    return addVect(fd, fs)

# Sphere intersection test adapted for CUDA
@cuda.jit(device=True)
def intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius):
    oc = diffVect(sphere_center, ray_origin)
    a = scalaire(ray_dir, ray_dir)
    b = 2.0 * scalaire(oc, ray_dir)
    c = scalaire(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return -1  # No intersection
    else:
        return (-b - math.sqrt(discriminant)) / (2.0 * a)

@cuda.jit(device=True)
def texture(o, lum, objColor, vert_pos, center, light_color, F0, roughness):

    color = (0.0, 0.0, 0.0)  # Initialize color for this path
    n = normalize(diffVect(center, vert_pos))  # Surface normal
    viewVect = normalize(diffVect(vert_pos, o))  # View vector

    # Light vector and half vector calculations
    l = normalize(diffVect(vert_pos, lum))
    h = normalize(addVect(viewVect, l))
    # Calculate color contribution from this sample
    color = multiVect(croi(BRDF(F0, n, h, viewVect, l, roughness, objColor), multiVect(light_color,1/(norm(l)**2))),max(0, scalaire(n, l)))
    return color  # Return accumulated color

#convert color from vecteur3 [0;1] to [0;255]
@cuda.jit(device=True)
def convert_color_to_rgb(col):
    return (max(0, min(255, col[0]*255)), max(0, min(255, col[1]*255)), max(0, min(255, col[2]*255)))

# Kernel for rendering scene
@cuda.jit
def render_kernel(pixels, width, height, camera_pos, sphere_centers, sphere_radii, num_obj, sample, state, emission, objColor, F0, roughness):
    x, y = cuda.grid(2)
    if x < width and y < height:
        hit = False
        color = (0, 0, 0)
        #rayon li ansifto
        ray_dir = normalize((x - width / 2, y - height / 2, -500 - camera_pos[2]))
        closest_t = 1000000
        for i in range(num_obj):# check collision 
            t = intersect_sphere(camera_pos, ray_dir, sphere_centers[i], sphere_radii[i])
            if 0 < t < closest_t:
                closest_t = t
                hit = True
                index = i
        if hit and emission[index]: pixels[y, x] = convert_color_to_rgb(objColor[index]) #collision avec source lumineuse
        elif hit: # collision with object
            vert_pos = (camera_pos[0] + closest_t * ray_dir[0], camera_pos[1] + closest_t * ray_dir[1], camera_pos[2] + closest_t * ray_dir[2])
            n = normalize(diffVect(sphere_centers[index] ,vert_pos))
            colorRef2 = (0, 0, 0)
            for ray1 in range(sample):  # Sample for reflections (envoie dautre rayon)
                # Generate a random reflection vector
                theta = xoroshiro128p_uniform_float32(state, x*ray1) * 2 * math.pi
                phi = xoroshiro128p_uniform_float32(state, y*ray1) * math.pi / 2
                r = (math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi))  #roundom[y,ray1]
                new_d = (n[0] - r[0], n[1] - r[1], n[2] - r[2])  # New direction after reflection
                closest_t2 = 100000
                hit2 = False
                # emet sample-rayon 
                for i in range(num_obj):
                    t2 = intersect_sphere(vert_pos, new_d, sphere_centers[i], sphere_radii[i])
                    if 0 < t2 < closest_t2:
                        closest_t2 = t2
                        hit2 = True
                        index2 = i
                color2 = (0, 0, 0)

                # check if the ray hits a light source
                if hit2 and emission[index2]:
                    vert_hit = (vert_pos[0] + closest_t2 * new_d[0], vert_pos[1] + closest_t2 * new_d[1], vert_pos[2] + closest_t2 * new_d[2])
                    color2 = texture(camera_pos, vert_hit, objColor[index], vert_pos, sphere_centers[index], objColor[index2], F0[index], roughness[index2])#objColor[index2]'''
                elif hit2: 
                    vert_hit = (vert_pos[0] + closest_t2 * new_d[0], vert_pos[1] + closest_t2 * new_d[1], vert_pos[2] + closest_t2 * new_d[2])
                    colorRef = (0, 0, 0)

                    #emet rayon encore
                    for ray2 in range(sample):
                        hit3 = False
                        theta = xoroshiro128p_uniform_float32(state, x*ray2) * 2 * math.pi
                        phi = xoroshiro128p_uniform_float32(state, y*ray2) * math.pi / 2
                        r = (math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi))  #roundom[x,ray2]
                        n_r = normalize(diffVect(sphere_centers[index2] ,vert_hit))
                        new_d2 = (n_r[0] - r[0], n_r[1] - r[1], n_r[2] - r[2])  # nouvelle direction after reflection
                        closest_t3 = 100000
                        for i in range(num_obj):
                            t3 = intersect_sphere(vert_hit, new_d2, sphere_centers[i], sphere_radii[i])
                            if 0 < t3 < closest_t3:
                                closest_t3 = t3
                                hit3 = True
                                index3 = i
                        color1 = (0, 0, 0)
                        if hit3 and emission[index3]:
                            vert_lum = (vert_hit[0] + closest_t3 * new_d2[0], vert_hit[1] + closest_t3 * new_d2[1], vert_hit[2] + closest_t3 * new_d2[2])
                            color1 = texture(vert_pos, vert_lum, objColor[index2], vert_hit, sphere_centers[index2], objColor[index3], F0[index2], roughness[index2])
                        
                        colorRef = addVect(colorRef, color1)

                    colorRef = multiVect(colorRef, 1/(2*math.pi*sample))
                    color2 = texture(camera_pos, vert_hit, objColor[index], vert_pos, sphere_centers[index], colorRef, F0[index], roughness[index])

                colorRef2 = addVect(color2, colorRef2)

            color = multiVect(colorRef2, 1/(sample))  
            #color = texture(camera_pos, (0,0,-300), objColor[index], vert_pos, sphere_centers[index], (1.0,1.0,1.0))
            pixels[y, x] = convert_color_to_rgb(color) #(xoroshiro128p_uniform_float32(state, x), xoroshiro128p_uniform_float32(state, y), xoroshiro128p_uniform_float32(state, x*y)) #color

# Main function to setup scene and call the rendering kernel
def render_scene():
    width, height = 300, 300
    pixels = np.zeros((height, width, 3), dtype=np.float32)

    d_pixels = cuda.to_device(pixels)

    num_obj = 4
    camera_pos = (0.0, 0.0, -1300.0)
    sphere_center = ((0.0, -200.0, 0.0),(0.0, 200.0, 0.0),(-300.0, 0.0, -400.0),(20000.0, 0.0, 0.0))
    sphere_radius = (150.0 ,150.0, 200.0, 19850.0)
    isEmissive = (False, False, True, False)
    objColor = ((120/255, 0.0, 1.0), (1.0, 0.0, 0.0), (10.0, 10.0, 10.0), (10.0, 10.0, 0.0))
    F0_r = ((0.7, 0.7, 0.7), (0.9, 0.9, 0.9), (0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
    roughness = (0.4, 0.1, 0.0, 0.5)

    sample = 10

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    state = create_xoroshiro128p_states(sample*width*height, seed=1257)
    #roundom = np.array([[(math.sin(uniform(0, math.pi/2)) * math.cos(uniform(0, 2*math.pi)), math.sin(uniform(0, math.pi/2)) * math.sin(uniform(0, 2*math.pi)), math.cos(uniform(0, math.pi/2))) for _ in range(width)] for _ in range(height) ])
    #d_roundom = cuda.to_device(roundom)


    render_kernel[blockspergrid, threadsperblock](d_pixels, width, height, camera_pos, sphere_center, sphere_radius, num_obj, sample, state, isEmissive, objColor, F0_r, roughness)

    pixels = d_pixels.copy_to_host()
    screen = pygame.display.set_mode((width, height))

    pygame.surfarray.blit_array(screen, pixels)
    
    running = True
    while running :

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.update()
    pygame.quit()
            

if __name__ == "__main__":
    render_scene()
