import numpy as np
import random
import h5py
from scipy.io import savemat
from scipy.ndimage import binary_dilation
from deepsubspacemri.sim.dwi import rotTensorAroundX as RTAX
from deepsubspacemri.sim.dwi import rotTensorAroundY as RTAY
from deepsubspacemri.sim.dwi import rotTensorAroundZ as RTAZ
from deepsubspacemri.sim import dwi

def generate_random_angle():
    while True:
        # Generate a random number between 0 and 90
        number = random.randint(0, 90)
        # Check if the number is divisible by 15
        if number % 15 == 0:
            return number

def create_circle_outline(height, width, radius, baseTensor, d):
    # Create a grid of coordinates
    y, x = np.ogrid[0:height+1, 0:width+1]
    
    # Calculate distance from each point to the center
    distance_from_center = np.sqrt((x - width/2)**2 + (y - height/2)**2)
    
    # Create a binary mask where points inside the circle are set to 1
    circle_mask = distance_from_center <= radius
    
    # Dilate the circle by one pixel to get the outline
    circle_dilated = binary_dilation(circle_mask)
    
    # Subtract the dilated circle from the original circle to get the outline
    circle_outline = circle_dilated ^ circle_mask

    grid = np.zeros((6,height+1, width+1))

    angle_step = 90/(radius+1)
    for row in range(grid.shape[1]):
        for col in range(grid.shape[2]):
            if circle_outline[col, row] == 1:
                sign = np.where(int(col>width//2), 1, -1)
                grid[:,col, row] = RTAZ(baseTensor,row*angle_step*sign) * d

    
    return grid

def create_circle_outline_t2(height, width, radius, t2_weighting):
    # Create a grid of coordinates
    y, x = np.ogrid[0:height+1, 0:width+1]
    
    # Calculate distance from each point to the center
    distance_from_center = np.sqrt((x - width/2)**2 + (y - height/2)**2)
    
    # Create a binary mask where points inside the circle are set to 1
    circle_mask = distance_from_center <= radius
    
    # Dilate the circle by one pixel to get the outline
    circle_dilated = binary_dilation(circle_mask)
    
    # Subtract the dilated circle from the original circle to get the outline
    circle_outline = circle_dilated ^ circle_mask

    grid = np.zeros((height+1, width+1))

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if circle_outline[col, row] == 1:
                grid[col, row] = abs(t2_weighting)
    return grid

def mark_outer_part(matrix, value):
    rows = len(matrix)
    cols = len(matrix[0])
    point_right = False
    for i in range(rows):
        for j in range(cols):
            if j < cols-1:
                if matrix[i][j] == 0 and matrix[i][j+1] == 0:
                    matrix[i][j] = value
                elif matrix[i][j] == 0:
                    matrix[i][j] = value
                    point_right = True
                    break
            else:
                matrix[i][j] = value
        if point_right:
            for j in range(cols-1, 0, -1):
                if matrix[i][j] == 0 and matrix[i][j-1] == 0:
                    matrix[i][j] = value
                elif matrix[i][j] == 0:
                    matrix[i][j] = value
                    point_right = False
                    break

def replace_values(matrix, valueToReplace, Tensor, phase=False):
    matrix2 = np.zeros_like(matrix)
    rows = len(matrix[-2])
    cols = len(matrix[-1])
    if len(matrix.shape) ==3:
        for i in range(rows):
            for j in range(cols):
                if matrix[0,i,j] == valueToReplace:
                    if type(Tensor) == int:
                        matrix[0,i,j] = Tensor
                    else:
                        rotation = 15*(j//15)
                        rotation2 = 15*(i//5)
                        weight = i%8<3
                        fillTensor = RTAZ(Tensor, rotation)
                        d1 = 2e-3
                        d2 = 2e-3
                        matrix[:,i,j] = fillTensor * d1
                        matrix2[:,i,j] = RTAX(TensorY, rotation2) * weight * d2
    else:
        for i in range(rows):
            for j in range(cols):
                if matrix[i,j] == valueToReplace:
                    if phase:
                        matrix[i,j] = random.uniform(-np.pi,np.pi)
                    else:
                        matrix[i,j] = Tensor
    return matrix2


def add_noise(x_clean, scale, noiseType = 'gaussian'):

    if noiseType== 'gaussian':
        x_noisy = x_clean + np.random.normal(loc = 0,
                                            scale = scale,
                                            size=x_clean.shape)
    elif noiseType== 'rician':
        noise1 =np.random.normal(0, scale, size=x_clean.shape)
        noise2 = np.random.normal(0, scale, size=x_clean.shape)
        x_noisy = np.sqrt((x_clean + noise1) ** 2 + noise2 ** 2)

    x_noisy[x_noisy < 0.] = 0.
    x_noisy[x_noisy > 1.] = 1.

    return x_noisy

def create_EPI_phantom(basic_phantom):
    #usage of table for T1, T2, PD from aubert_broche

    angle_tissue = 0.5
    angle_fat = 2.0
    bone = [0.01,1,1,angle_tissue] #PD,T1,T2
    CSF = [1,2569,329,angle_tissue]
    whiteMatter = [0.77,500,70,angle_tissue]
    fat = [1,350,70,angle_fat]
    grayMatter = [0.86,833,83,angle_tissue]
    air = [0.0001,1,1,angle_tissue]
    

    for timingConst in range(basic_phantom.shape[0]-1):
    
        valueBone = bone[timingConst]

        basic_phantom[timingConst, 39:62, 4] = valueBone
        basic_phantom[timingConst, 39:62, 5] = valueBone
        basic_phantom[timingConst, 39:62, 6] = valueBone
        basic_phantom[timingConst, 39:62, 7] = valueBone
        basic_phantom[timingConst, 39:62, 8] = valueBone

        for i in range(0, int(90/5),1):
            basic_phantom[timingConst, 38-i, 4+i] = valueBone
            basic_phantom[timingConst, 38-i, 5+i] = valueBone
            basic_phantom[timingConst, 38-i, 6+i] = valueBone

            basic_phantom[timingConst, 38-i, 7+i] = valueBone
            basic_phantom[timingConst, 38-i, 8+i] = valueBone
            
            basic_phantom[timingConst, 62+i, 4+i] = valueBone
            basic_phantom[timingConst, 62+i, 5+i] = valueBone
            basic_phantom[timingConst, 62+i, 6+i] = valueBone

            basic_phantom[timingConst, 62+i, 7+i] = valueBone
            basic_phantom[timingConst, 62+i, 8+i] = valueBone

        basic_phantom[timingConst, 21, 22:78] = valueBone
        basic_phantom[timingConst, 79, 22:78] = valueBone

        basic_phantom[timingConst, 22, 22:78] = valueBone
        basic_phantom[timingConst, 78, 22:78] = valueBone

        basic_phantom[timingConst, 23, 22:78] = valueBone
        basic_phantom[timingConst, 77, 22:78] = valueBone

        basic_phantom[timingConst, 24, 22:78] = valueBone
        basic_phantom[timingConst, 76, 22:78] = valueBone

        basic_phantom[timingConst, 25, 22:78] = valueBone
        basic_phantom[timingConst, 75, 22:78] = valueBone

        basic_phantom[timingConst, 39:62, 99 - 4] = valueBone
        basic_phantom[timingConst, 39:62, 99 - 5] = valueBone
        basic_phantom[timingConst, 39:62, 99 - 6] = valueBone
        basic_phantom[timingConst, 39:62, 99 - 7] = valueBone
        basic_phantom[timingConst, 39:62, 99 - 8] = valueBone


        for i in range(0, int(90/5),1):
            basic_phantom[timingConst, 38 - i, 99 - 4 - i] = valueBone
            basic_phantom[timingConst, 38 - i, 99 - 5 - i] = valueBone
            basic_phantom[timingConst, 38 - i, 99 - 6 - i] = valueBone

            basic_phantom[timingConst, 38 - i, 99 - 7 - i] = valueBone
            basic_phantom[timingConst, 38 - i, 99 - 8 - i] = valueBone
            
            basic_phantom[timingConst, 62 + i, 99 - 4 - i] = valueBone
            basic_phantom[timingConst, 62 + i, 99 - 5 - i] = valueBone
            basic_phantom[timingConst, 62 + i, 99 - 6 - i] = valueBone

            basic_phantom[timingConst, 62 + i, 99 - 7 - i] = valueBone
            basic_phantom[timingConst, 62 + i, 99 - 8 - i] = valueBone

        #eyes 

        valueEye1 = CSF[timingConst] #CSF
        maxRad = 7
        cornerX = 60
        cornerY = 30
        gridSize = 2*maxRad+3
        for i in range(maxRad):
            basic_phantom[timingConst, cornerY:cornerY+gridSize, cornerX:cornerX+gridSize] += create_circle_outline_t2(2*maxRad+2, 2*maxRad+2, maxRad-i, valueEye1)
        basic_phantom[timingConst, cornerY:cornerY+gridSize, cornerX:cornerX+gridSize] += create_circle_outline_t2(2*maxRad+2, 2*maxRad+2, 0, valueEye1)

        valueEye2 = whiteMatter[timingConst]# white_matter
        maxRad = 7
        cornerX = 60
        cornerY = 55
        gridSize = 2*maxRad+3
        for i in range(maxRad):
            basic_phantom[timingConst, cornerY:cornerY+gridSize, cornerX:cornerX+gridSize] += create_circle_outline_t2(2*maxRad+2, 2*maxRad+2, maxRad-i, valueEye2)
        basic_phantom[timingConst, cornerY:cornerY+gridSize, cornerX:cornerX+gridSize] += create_circle_outline_t2(2*maxRad+2, 2*maxRad+2, 0, valueEye2)

        #Mouth
        valueMouth = whiteMatter[timingConst]# white_matter
        start = 32
        end = 69
        y_start = 39
        angle_step = 10
        angle_step2 = 15    
        cond = False
        y= 0
        basic_phantom[timingConst, start:end+1, 41] = valueMouth
        basic_phantom[timingConst, start:end+1, 40] = valueMouth
        for l in range(150//angle_step2):
            for i in range((end-start)//2):
                if angle_step*i > (90-l*angle_step):
                    if cond == False:
                        y = i
                        cond = True
                else:
                    y = i
                basic_phantom[timingConst, start + i+l, y_start-y] = valueMouth
                basic_phantom[timingConst, start + i+l, y_start-1-y] = valueMouth

                basic_phantom[timingConst, end - i -l, y_start-y] = valueMouth
                basic_phantom[timingConst, end - i-l, y_start-1-y] = valueMouth


        basic_phantom[timingConst, start + (end-start)//2, 40-11] = valueMouth
        basic_phantom[timingConst, start + (end-start)//2, 40-12] = valueMouth
        basic_phantom[timingConst, start + (end-start)//2+1, 40-11] = valueMouth
        basic_phantom[timingConst, start + (end-start)//2+1, 40-12] = valueMouth

        #nose
        #start at y = 50, x = 49,50, height = 15

        noseValue = fat[timingConst]#fat

        basic_phantom[timingConst, 49:50+1, 64] = noseValue
        basic_phantom[timingConst, 49:50+1, 63] = noseValue

        angle_step = 15
        for i in range(13):
            if i <= 2:
                basic_phantom[timingConst, 49-(i//2), 62-i] = noseValue
                basic_phantom[timingConst, 50+(i//2), 62-i] = noseValue
            elif i <= 8:
                if i%2 == 0:
                    basic_phantom[timingConst, 49-(i//2),  62-i] = noseValue
                    basic_phantom[timingConst, 50+(i//2), 62-i] = noseValue
                else:
                    basic_phantom[timingConst, 49-(i//2),  62-i] = noseValue
                    basic_phantom[timingConst, 50+(i//2), 62-i] = noseValue
            elif i <= 11:
                basic_phantom[timingConst, 49-(i//2), 62-i] = noseValue
                basic_phantom[timingConst, 50+(i//2), 62-i] = noseValue
            else:
                basic_phantom[timingConst, 49-(i//2)+1, 62-i] =noseValue
                basic_phantom[timingConst, 50+(i//2)-1, 62-i] = noseValue

        x_l = 49
        x_r =  50
        y_start = 59
        for i in range(11):
            width = x_r-x_l+1
            #left
            for l in range(width//2):
                basic_phantom[timingConst, x_l+l, y_start-i] = noseValue
            #right
            for l in range(width//2):
                basic_phantom[timingConst, x_r-l, y_start-i] = noseValue
            if i%2 == 0:
                x_r +=1
                x_l -=1

        basic_phantom[timingConst, 45:54+1, 50] = noseValue

        #rest inner part
        innerPartValue = grayMatter[timingConst] #gray matter
        mark_outer_part(basic_phantom[timingConst,:,:], -2)
        basic_phantom_other_tensor = replace_values(basic_phantom[timingConst,:,:], 0, innerPartValue)
        if timingConst == 3:
            replace_values(basic_phantom[timingConst,:,:], -2, air[timingConst], phase=True)
        else:
            replace_values(basic_phantom[timingConst,:,:], -2, air[timingConst])
    


Tensor0 = np.array([0, 0, 0, 0, 0, 0])
TensorX = np.array([1, 0, 0.5, 0, 0, 0.5])
TensorY = np.array([0.5, 0, 1, 0, 0, 0.5])
TensorZ = np.array([0.5, 0, 0.5, 0, 0, 1])
TensorISO = np.array([1, 0, 1, 0, 0, 1])

basic_phantom = np.zeros((6,100,100))
basic_phantom_EPI = np.zeros((5,100,100), dtype=np.complex_) #SD, T1, T2, phase, combined

dOuterBorder = 2e-3
dMiddle1 = 1e-3
dMiddle2 = 1.5e-3
dMiddle3 = 2.5e-3
dInner = 3e-3
# dOuterBorder = 1
# dMiddle1 = 1
# dMiddle2 = 1
# dMiddle3 = 1
# dInner = 1
basic_phantom[:, 39:62, 4] = np.tile(TensorX.reshape(6,1), (1, 23)) * dOuterBorder
basic_phantom[:, 39:62, 5] = np.tile(TensorX.reshape(6,1), (1, 23)) * dMiddle1
basic_phantom[:, 39:62, 6] = np.tile(TensorX.reshape(6,1), (1, 23)) * dMiddle2
basic_phantom[:, 39:62, 7] = np.tile(RTAX(TensorY,-60).reshape(6,1), (1, 23)) * dMiddle3
basic_phantom[:, 39:62, 8] = np.tile(RTAX(TensorY,-30).reshape(6,1), (1, 23)) * dInner

for i in range(0, int(90/5),1):
    basic_phantom[:, 38-i, 4+i] = RTAZ(TensorX, 5*i) * dOuterBorder
    basic_phantom[:, 38-i, 5+i] = RTAZ(TensorX, 5*i) * dMiddle1
    basic_phantom[:, 38-i, 6+i] = RTAZ(TensorX, 5*i) * dMiddle2

    basic_phantom[:, 38-i, 7+i] = RTAZ(RTAX(TensorY,-60),-45) * dMiddle3
    basic_phantom[:, 38-i, 8+i] = RTAZ(RTAX(TensorY,-30),-45) * dInner
    
    basic_phantom[:, 62+i, 4+i] = RTAZ(TensorX, -5*i) * dOuterBorder
    basic_phantom[:, 62+i, 5+i] = RTAZ(TensorX, -5*i) * dMiddle1
    basic_phantom[:, 62+i, 6+i] = RTAZ(TensorX, -5*i) * dMiddle2

    basic_phantom[:, 62+i, 7+i] = RTAZ(RTAX(TensorY,-60),45) * dMiddle3
    basic_phantom[:, 62+i, 8+i] = RTAZ(RTAX(TensorY,-30),45) * dInner

basic_phantom[:, 21, 22:78] = np.tile(RTAZ(TensorX, 90).reshape(6,1), (1,56)) * dOuterBorder
basic_phantom[:, 79, 22:78] = np.tile(RTAZ(TensorX, 90).reshape(6,1), (1,56)) * dOuterBorder

basic_phantom[:, 22, 22:78] = np.tile(RTAZ(TensorX, 90).reshape(6,1), (1,56)) * dMiddle1
basic_phantom[:, 78, 22:78] = np.tile(RTAZ(TensorX, 90).reshape(6,1), (1,56)) * dMiddle1

basic_phantom[:, 23, 22:78] = np.tile(RTAZ(TensorX, 90).reshape(6,1), (1,56)) * dMiddle2
basic_phantom[:, 77, 22:78] = np.tile(RTAZ(TensorX, 90).reshape(6,1), (1,56)) * dMiddle2 

basic_phantom[:, 24, 22:78] = np.tile(RTAY(TensorX,60).reshape(6,1), (1, 56)) * dMiddle3
basic_phantom[:, 76, 22:78] = np.tile(RTAY(TensorX,-60).reshape(6,1), (1, 56)) * dMiddle3

basic_phantom[:, 25, 22:78] = np.tile(RTAY(TensorX,30).reshape(6,1), (1, 56)) * dInner
basic_phantom[:, 75, 22:78] = np.tile(RTAY(TensorX,-30).reshape(6,1), (1, 56)) * dInner

basic_phantom[:, 39:62, 99 - 4] = np.tile(TensorX.reshape(6,1), (1, 23)) * dOuterBorder
basic_phantom[:, 39:62, 99 - 5] = np.tile(TensorX.reshape(6,1), (1, 23)) * dMiddle1
basic_phantom[:, 39:62, 99 - 6] = np.tile(TensorX.reshape(6,1), (1, 23)) * dMiddle2
basic_phantom[:, 39:62, 99 - 7] = np.tile(RTAX(TensorY,60).reshape(6,1), (1, 23)) * dMiddle3
basic_phantom[:, 39:62, 99 - 8] = np.tile(RTAX(TensorY,30).reshape(6,1), (1, 23)) * dInner


for i in range(0, int(90/5),1):
    basic_phantom[:, 38 - i, 99 - 4 - i] = RTAZ(TensorX, 5*i) * dOuterBorder
    basic_phantom[:, 38 - i, 99 - 5 - i] = RTAZ(TensorX, 5*i) * dMiddle1
    basic_phantom[:, 38 - i, 99 - 6 - i] = RTAZ(TensorX, 5*i) * dMiddle2

    basic_phantom[:, 38 - i, 99 - 7 - i] = RTAZ(RTAX(TensorY,60),45) * dMiddle3
    basic_phantom[:, 38 - i, 99 - 8 - i] = RTAZ(RTAX(TensorY,30),45) * dInner
    
    basic_phantom[:, 62 + i, 99 - 4 - i] = RTAZ(TensorX, -5*i) * dOuterBorder
    basic_phantom[:, 62 + i, 99 - 5 - i] = RTAZ(TensorX, -5*i) * dMiddle1
    basic_phantom[:, 62 + i, 99 - 6 - i] = RTAZ(TensorX, -5*i) * dMiddle2

    basic_phantom[:, 62 + i, 99 - 7 - i] = RTAZ(RTAX(TensorY,60),-45) * dMiddle3
    basic_phantom[:, 62 + i, 99 - 8 - i] = RTAZ(RTAX(TensorY,30),-45) * dInner

#eyes 

dEye1 = 2e-3
maxRad = 7
cornerX = 60
cornerY = 30
gridSize = 2*maxRad+3
for i in range(maxRad):
    basic_phantom[:, cornerY:cornerY+gridSize, cornerX:cornerX+gridSize] += create_circle_outline(2*maxRad+2, 2*maxRad+2, maxRad-i, RTAX(TensorY, i*15), dEye1)
basic_phantom[:, cornerY:cornerY+gridSize, cornerX:cornerX+gridSize] += create_circle_outline(2*maxRad+2, 2*maxRad+2, 0, TensorISO, dEye1)

dEye2 = 1e-3
maxRad = 7
cornerX = 60
cornerY = 55
gridSize = 2*maxRad+3
for i in range(maxRad):
    basic_phantom[:, cornerY:cornerY+gridSize, cornerX:cornerX+gridSize] += create_circle_outline(2*maxRad+2, 2*maxRad+2, maxRad-i, RTAX(TensorY, -i*15), dEye2)
basic_phantom[:, cornerY:cornerY+gridSize, cornerX:cornerX+gridSize] += create_circle_outline(2*maxRad+2, 2*maxRad+2, 0, TensorISO, dEye2)

#Mouth
dMouth = 2.5e-3
start = 32
end = 69
y_start = 39
angle_step = 10
angle_step2 = 15
cond = False
y= 0
basic_phantom[:, start:end+1, 41] = np.tile(TensorX.reshape(6,1), (1,end-start+1)) * dMouth
basic_phantom[:, start:end+1, 40] = np.tile(TensorX.reshape(6,1), (1,end-start+1)) * dMouth
for l in range(150//angle_step2):
    for i in range((end-start)//2):
        if angle_step*i > (90-l*angle_step):
            if cond == False:
                y = i
                cond = True
        else:
            y = i
        basic_phantom[:, start + i+l, y_start-y] = RTAX(RTAZ(TensorY, y*angle_step), l*angle_step2)  * dMouth
        basic_phantom[:, start + i+l, y_start-1-y] = RTAX(RTAZ(TensorY, y*angle_step), l*angle_step2) * dMouth

        basic_phantom[:, end - i -l, y_start-y] = RTAX(RTAZ(TensorY, -y*angle_step), l*angle_step2) * dMouth
        basic_phantom[:, end - i-l, y_start-1-y] = RTAX(RTAZ(TensorY, -y*angle_step), l*angle_step2) * dMouth


basic_phantom[:, start + (end-start)//2, 40-11] = TensorX * dMouth
basic_phantom[:, start + (end-start)//2, 40-12] = TensorX * dMouth
basic_phantom[:, start + (end-start)//2+1, 40-11] = TensorX * dMouth
basic_phantom[:, start + (end-start)//2+1, 40-12] = TensorX * dMouth

#nose
#start at y = 50, x = 49,50, height = 15

dNose = 3e-3

basic_phantom[:, 49:50+1, 64] = np.tile(TensorY.reshape(6,1), (1,2)) * dNose
basic_phantom[:, 49:50+1, 63] = np.tile(TensorY.reshape(6,1), (1,2)) * dNose

angle_step = 15
for i in range(13):
    if i <= 2:
        basic_phantom[:, 49-(i//2), 62-i] = RTAZ(TensorY, -i*angle_step) * dNose
        basic_phantom[:, 50+(i//2), 62-i] = RTAZ(TensorY, i*angle_step) * dNose
    elif i <= 8:
        if i%2 == 0:
            basic_phantom[:, 49-(i//2),  62-i] = RTAZ(TensorY, -30) * dNose
            basic_phantom[:, 50+(i//2), 62-i] = RTAZ(TensorY, 30) * dNose
        else:
            basic_phantom[:, 49-(i//2),  62-i] = TensorY * dNose
            basic_phantom[:, 50+(i//2), 62-i] = TensorY * dNose
    elif i <= 11:
        basic_phantom[:, 49-(i//2), 62-i] = RTAZ(TensorY, -30+(i%9)*angle_step) * dNose
        basic_phantom[:, 50+(i//2), 62-i] = RTAZ(TensorY, 30-(i%9)*angle_step) * dNose
    else:
        basic_phantom[:, 49-(i//2)+1, 62-i] =RTAZ(TensorY,45) * dNose
        basic_phantom[:, 50+(i//2)-1, 62-i] = RTAZ(TensorY,-45) * dNose

x_l = 49
x_r =  50
y_start = 59
for i in range(11):
    width = x_r-x_l+1
    #left
    for l in range(width//2):
        basic_phantom[:, x_l+l, y_start-i] = RTAY(RTAZ(basic_phantom[:, x_l+l-1, y_start-i], 90),-l*15)
    #right
    for l in range(width//2):
        basic_phantom[:, x_r-l, y_start-i] = RTAY(RTAZ(basic_phantom[:, x_r-l+1, y_start-i], 90),l*15)
    if i%2 == 0:
        x_r +=1
        x_l -=1

basic_phantom[:, 45:54+1, 50] = np.tile(TensorX.reshape(6,1), (1,10)) * dNose

mark_outer_part(basic_phantom[0,:,:], -2)
basic_phantom_other_tensor = replace_values(basic_phantom, 0, TensorX)
replace_values(basic_phantom, -2, 0)

# not to use with diffusivities
mdic = {"matrix1": basic_phantom,"matrix2": basic_phantom_other_tensor, "label": "phantom"}
savemat('matlab_matrix.mat', mdic)

# create q-space images
create_EPI_phantom(basic_phantom_EPI)
print(basic_phantom_EPI.shape)
#T2 weighting
TE = 80
TR = 5000
t2_weighted = basic_phantom_EPI[0]*(1 - 2 *np.exp(-(TR-TE/2)/basic_phantom_EPI[1])+ np.exp(-TR/basic_phantom_EPI[1]))*np.exp(-TE/basic_phantom_EPI[2])
print(t2_weighted)
print(t2_weighted.shape)
basic_phantom_EPI[-1,:,:] = t2_weighted

g_and_b_file = h5py.File('/home/hpc/iwbi/iwbi019h/DeepSubspaceMRI/deepsubspacemri/data_files/3shell_126dir_diff_encoding.h5', 'r')

#f = h5py.File(ACQ_DIR + '/3shell_126dir_diff_encoding.h5', 'r')

bvals = g_and_b_file['bvals'][:]
bvals = bvals[:, np.newaxis]
bvecs = g_and_b_file['bvecs'][:]

g_and_b_file.close()

q_space_grid = np.zeros((126, 100, 100))
for i in range(q_space_grid.shape[1]):
    for j in range(q_space_grid.shape[2]):
        if basic_phantom_other_tensor[0,i,j] == 0:
            q_space_grid[:,i,j] = dwi.calc_DTI_res(bvals, bvecs, basic_phantom[:,i,j],1)
        else:
            frac = 0.5
            q_space_grid[:,i,j] = dwi.calc_DTI_res(bvals, bvecs, basic_phantom[:,i,j],1)*frac + dwi.calc_DTI_res(bvals, bvecs, basic_phantom_other_tensor[:,i,j],1) * (1-frac)
    


f = h5py.File('basic_phantom_126dir.h5','w')
f.create_dataset('combined_q_space', data=q_space_grid)
f.create_dataset('phantom1_diffusion', data=basic_phantom)
f.create_dataset('phantom2_diffusion', data=basic_phantom_other_tensor)
f.create_dataset('EPI_phantom', data=basic_phantom_EPI)

# add noise
SNRs = [5, 10, 20, 35, 50]

for SNR in SNRs:
    g_space_grid_noised = add_noise(q_space_grid, 1/SNR, noiseType = 'rician')
    f.create_dataset('q_space_noised_SNR_' + str(SNR), data=g_space_grid_noised)



f.close()
