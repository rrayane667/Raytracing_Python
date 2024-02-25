import numpy as np
from numba import cuda, float32, int32
import math
import pygame

# Util functions adapted for CUDA, avoiding dependencies on external libraries
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
    return [(X[0] - 2*scal*N[0]), (X[1] - 2*scal*N[1]), (X[2] - 2*scal*N[2])]

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
    f_c = (1 - scalaire(N, H)) ** 5
    return (F0[0] + (1 - F0[0]) * f_c, F0[1] + (1 - F0[1]) * f_c, F0[2] + (1 - F0[2]) * f_c)

@cuda.jit(device=True)
def Flamb(c): #vecteur
    return multiVect(c, 1/math.pi)

@cuda.jit(device=True)
def Fspe( N, H, V, L, roughness): #scalaire
    return D(roughness, N, H)*G(N, V, L, roughness)/(4*scalaire(V, N)*scalaire(N, L) + 0.00001)

@cuda.jit(device=True)
def BRDF(F0, N, H, V, L, roughness, c):
    Ks = F(F0, N, H)
    Kd = diffVect(Ks, [1, 1, 1])
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

# Kernel for rendering scene
@cuda.jit
def render_kernel(pixels, width, height, camera_pos, sphere_center, sphere_radius):
    x, y = cuda.grid(2)
    if x < width and y < height:
        ray_dir = normalize((x - width / 2, y - height / 2, -500-camera_pos[2]))
        t = intersect_sphere(camera_pos, ray_dir, sphere_center, sphere_radius)
        if t > 0:
            pixels[y, x, 0] = 120  # Red channel
            pixels[y, x, 1] = 0    # Green channel
            pixels[y, x, 2] = 255  # Red channel
        else:
            pixels[y, x, :] = 0

# Main function to setup scene and call the rendering kernel
def render_scene():
    width, height = 300, 300
    pixels = np.zeros((height, width, 3), dtype=np.float32)

    d_pixels = cuda.to_device(pixels)

    camera_pos = (0, 0, -1300)
    sphere_center = (0, 0, 0)
    sphere_radius = 100

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    render_kernel[blockspergrid, threadsperblock](d_pixels, width, height, camera_pos, sphere_center, sphere_radius)

    pixels = d_pixels.copy_to_host()
    screen = pygame.display.set_mode((width, height))
    for i in range(width):
        for j in range(height):
            screen.set_at((i, j), pixels[i][j])
    
    running = True
    while running :

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.update()
    pygame.quit()
            

if __name__ == "__main__":
    render_scene()
