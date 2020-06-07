import taichi as ti
#ti.init(ti.cpu, debug=True)
ti.init(ti.cuda)

n=320
pixels=ti.var(dt=ti.f32,shape=(n*2,n))

RANDOM_TBL = ti.Vector(2, dt=ti.f32, shape=(n*2,n))

CellSize = 0.2

@ti.func
def rand1dT1d(t:ti.template)->ti.template:
    
    #ret = ti.sin(t*10+1.546)*143758.5453
    #return ti.abs(ret-int(ret))
    return RANDOM_TBL[t[0], t[1]]

@ti.func
def easeIn(t:ti.template)->ti.template:
    return t*t

@ti.func
def easeOut(t:ti.template)->ti.template:
    return 1-easeIn(1-t)

@ti.func
def easeInOut(t:ti.template)->ti.template:
    easeInValue = easeIn(t)
    easeOutValue = easeOut(t)
    return lerp(easeInValue, easeOutValue, t)
    #return 6*(t**5)-15*(t**4)+10*(t**3)

@ti.func
def lerp(a:ti.template, b:ti.template, t:ti.f32)->ti.template:
    return a*(1-t) + b*t

@ti.func
def gradientNoise(t:ti.template)->ti.f32:
    fraction = t-int(t)
    interpolator = easeInOut(fraction)

    cellNoiseZ = ti.Vector([0,0,0])

    #return fraction.dot(fraction)
    #return fraction[0]
    ft = int(t)
    ct = ti.cast(ti.ceil(t), ti.i32)

    lowerleftd = rand1dT1d(ti.Vector([ft[0], ft[1]]))*2-1
    

    #return (lowerleftd.dot(ti.Vector([1,1])))/2

    #return lowerleftd[1]

    lowerrightd = rand1dT1d(ti.Vector([ct[0], ft[1]]))*2-1
    upleftd = rand1dT1d(ti.Vector([ft[0], ct[1]]))*2-1
    uprightd = rand1dT1d(ti.Vector([ct[0], ct[1]]))*2-1


    lowleftv = lowerleftd.dot(fraction-ti.Vector([0, 0]))
    lowrightv = lowerrightd.dot(fraction-ti.Vector([1, 0]))

    #-return ti.abs(lowleftv)

    upleftv = upleftd.dot(fraction-ti.Vector([0, 1]))
    uprightv= uprightd.dot(fraction-ti.Vector([1, 1]))

    low = lerp(lowleftv, lowrightv, interpolator[0])
    #return low
    up = lerp(upleftv, uprightv, interpolator[0])

    #return low

    noise = lerp(low, up, interpolator[1])
    return noise


'''
    for z in ti.static(range(2)):
        cellNoiseY = ti.Vector([0,0,0])

        for y in ti.static(range(2)):
            cellNoiseX = ti.Vector([0,0,0])

            for x in ti.static(range(2)):
                cell = int(t) + ti.Vector([x, y, z])
                cellDirection = rand1dT1d(cell)*2-1
                compareVector = fraction - ti.Vector([x, y, z])
                cellNoiseX[x] = cellDirection.dot(compareVector)

            cellNoiseY[y] = lerp(cellNoiseX[0], cellNoiseX[1], interpolator[0])

        cellNoiseZ[z] = lerp(cellNoiseY[0], cellNoiseY[1], interpolator[1])

    noise = lerp(cellNoiseZ[0], cellNoiseZ[1], interpolator[2])
    return noise
'''
    #return previousLinePoint*interpolator + nextLinePoint*(1-interpolator)

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2-z[1]**2,z[1]*z[0]*2])
    
@ti.kernel
def paint(t:ti.f32):
    for i,j in pixels:#Parallized over all pixels
        #while z.norm()<20 and iterations<50:
        #    z=complex_sqr(z)+c
        #    iterations+=1
        pos = ti.Vector([i,j])
        pixels[i,j]= gradientNoise(pos/n / CellSize) #1-iterations*0.02

@ti.kernel
def generate_random():
    for i,j in pixels:
        RANDOM_TBL[i,j] = ti.Vector([ti.random(), ti.random()])

generate_random()

gui=ti.GUI("JuliaSet",res=(n*2,n))
for i in range(1000000):
    paint(i*0.3)
    gui.set_image(pixels)
    gui.show()