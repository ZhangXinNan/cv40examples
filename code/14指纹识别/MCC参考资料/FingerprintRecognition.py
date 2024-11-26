# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 07:56:15 2021

@author: Administrator
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

fingerprint = cv2.imread('sample_1_1.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("original",fingerprint)
cv2.imwrite("traditionalThoeryImage/original.bmp",fingerprint)
# show(fingerprint, f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}')
gx, gy = cv2.Sobel(fingerprint, cv2.CV_32F, 1, 0), cv2.Sobel(fingerprint, cv2.CV_32F, 0, 1)

cv2.imshow("gx",gx)
cv2.imshow("gy",gy)

gx2, gy2 = gx**2, gy**2
gm = np.sqrt(gx2 + gy2)

cv2.imshow("gx2",gx2)
cv2.imshow("gy2",gy2)
cv2.imshow("gm",gm)


sum_gm = cv2.boxFilter(gm, -1, (25, 25), normalize = False)
cv2.imshow("sum_gm",sum_gm)
# print("sum_gm",sum_gm)

thr = sum_gm.max() * 0.2
mask = cv2.threshold(sum_gm, thr, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
x= cv2.merge((mask, fingerprint, fingerprint))
cv2.imshow("original2",fingerprint)
cv2.imshow("mask",mask)
cv2.imshow("x",x)
cv2.imwrite("traditionalThoeryImage/fore.bmp",x)

W = (23, 23)#我们定义一个23x23的窗口
gxx = cv2.boxFilter(gx2, -1, W, normalize = False)# 在给定的滑动窗口大小下，对每个窗口内的像素值进行快速相加求和
gyy = cv2.boxFilter(gy2, -1, W, normalize = False)
gxy = cv2.boxFilter(gx * gy, -1, W, normalize = False) # gx * gy 
gxx_gyy = gxx - gyy
gxy2 = 2 * gxy

orientations = (cv2.phase(gxx_gyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction phase函数计算方向场
sum_gxx_gyy = gxx + gyy
strengths = np.divide(cv2.sqrt((gxx_gyy**2 + gxy2**2)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy!=0)#  计算置信度，也就是计算在W 中有多少梯度有同样的方向，自然数量越多，计算的结果越可靠
def draw_orientations(fingerprint, orientations, strengths, mask, scale = 3, step = 8, border = 0):
    if strengths is None:
        strengths = np.ones_like(orientations)
    h, w = fingerprint.shape
    sf = cv2.resize(fingerprint, (w*scale, h*scale), interpolation = cv2.INTER_NEAREST)
    res = cv2.cvtColor(sf, cv2.COLOR_GRAY2BGR)
    d = (scale // 2) + 1
    sd = (step+1)//2
    c = np.round(np.cos(orientations) * strengths * d * sd).astype(int)
    s = np.round(-np.sin(orientations) * strengths * d * sd).astype(int) # minus for the direction of the y axis
    thickness = 1 + scale // 5
    for y in range(border, h-border, step):
        for x in range(border, w-border, step):
            if mask is None or mask[y, x] != 0:
                ox, oy = c[y, x], s[y, x]
                cv2.line(res, (d+x*scale-ox,d+y*scale-oy), (d+x*scale+ox,d+y*scale+oy), (255,0,0), thickness, cv2.LINE_AA)
    return res
u=draw_orientations(fingerprint, orientations, strengths, mask, 1, 16)
cv2.imshow("u",u)
cv2.imwrite("traditionalThoeryImage/ridge.bmp",u)

region = fingerprint[10:90,80:130]
cv2.imshow("regioin",region)

# before computing the x-signature, the region is smoothed to reduce noise
smoothed = cv2.blur(region, (5,5), -1)
xs = np.sum(smoothed, 1) # the x-signature of the region
print(xs)
x = np.arange(region.shape[0])
f, axarr = plt.subplots(1,2, sharey = True)
axarr[0].imshow(region,cmap='gray')
axarr[1].plot(xs, x)
axarr[1].set_ylim(region.shape[0]-1,0)
plt.show()

# Find the indices of the x-signature local maxima
local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
x = np.arange(region.shape[0])
plt.plot(x, xs)
plt.xticks(local_maxima)
plt.grid(True, axis='x')
plt.show()
# Calculate all the distances between consecutive peaks
distances = local_maxima[1:] - local_maxima[:-1]
print(distances)
# [10 10 11 10  9  8  8]
# Estimate the ridge line period as the average of the above distances
ridge_period = np.average(distances)
print(ridge_period)# 9.428571428571429

_sigma_conv = (3.0/2.0)/((6*math.log(10))**0.5)
def _gabor_sigma(ridge_period):
    return _sigma_conv * ridge_period

def _gabor_size(ridge_period):
    p = int(round(ridge_period * 2 + 1))
    if p % 2 == 0:
        p += 1
    return (p, p)

def gabor_kernel(period, orientation):
    f = cv2.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi/2 - orientation, period, gamma = 1, psi = 0)
    f /= f.sum()
    f -= f.mean()
    return f
or_count = 8
gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi/or_count)]
print(gabor_bank)
for i in range(8):
    cv2.imshow("gabor_bank"+str(i),gabor_bank[i])


# Filter the whole image with each filter
# Note that the negative image is actually used, to have white ridges on a black background as a result
nf = 255-fingerprint
all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank])
'''
for i in range(8):
    cv2.imshow("all_filtered"+str(i),all_filtered[i])
'''


y_coords, x_coords = np.indices(fingerprint.shape)
# For each pixel, find the index of the closest orientation in the gabor bank
orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
# Take the corresponding convolution result for each pixel, to assemble the final result
filtered = all_filtered[orientation_idx, y_coords, x_coords]
# Convert to gray scale and apply the mask
enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
cv2.imshow("enhanced",enhanced)
cv2.imwrite("traditionalThoeryImage/enhanced.bmp",enhanced)
# Binarization
_, ridge_lines = cv2.threshold(enhanced, 32, 255, cv2.THRESH_BINARY)# enhanced 是增强之后的图像
# show(fingerprint, ridge_lines, cv.merge((ridge_lines, fingerprint, fingerprint)))

cv2.imshow("ridge_lines",ridge_lines)
xm=cv2.merge((ridge_lines, fingerprint, fingerprint))
cv2.imshow("xm",xm)
# print("xm",xm)

# Thinning
skeleton = cv2.ximgproc.thinning(ridge_lines, thinningType = cv2.ximgproc.THINNING_GUOHALL)
cv2.imshow("skeleton",skeleton)
cv2.imwrite("traditionalThoeryImage/skeleton.bmp",skeleton)
xs=cv2.merge((fingerprint, fingerprint, skeleton))
cv2.imshow("xs",xs)

def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))
# Create a filter that converts any 8-neighborhood into the corresponding byte value [0,255]
cn_filter = np.array([[  1,  2,  4],
                      [128,  0,  8],
                      [ 64, 32, 16]
                     ])
# Create a lookup table that maps each byte value to the corresponding crossing number
all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
print("all_8_neighborhoods",all_8_neighborhoods)
cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)
# np.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=10000)
print("cn_lut",cn_lut)
print("len(cn_lut)",len(cn_lut))
# Skeleton: from 0/255 to 0/1 values
skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
cv2.imshow("skeleton01",skeleton01*255)
# 应用设计好的filter，将8领域转换为0-255的byte
cn_values = cv2.filter2D(skeleton01, -1, cn_filter, borderType = cv2.BORDER_CONSTANT)
cv2.imshow("cn_values",cn_values)
print("cn_values",cn_values)
# 使用查找表，获取crossing numbers的值
cn = cv2.LUT(cn_values, cn_lut)
print("cn",cn)
# Keep only crossing numbers on the skeleton
cn[skeleton==0] = 0
print("cn2",cn)

minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]

print(minutiae)
# show(draw_minutiae(fingerprint, minutiae), skeleton, draw_minutiae(skeleton, minutiae))
def draw_minutiae(fingerprint, minutiae, termination_color = (255,0,0), bifurcation_color = (0,0,255)):
    res = cv2.cvtColor(fingerprint, cv2.COLOR_GRAY2BGR)
    
    for x, y, t, *d in minutiae:
        color = termination_color if t else bifurcation_color
        if len(d)==0:
            cv2.drawMarker(res, (x,y), color, cv2.MARKER_CROSS, 8)
        else:
            d = d[0]
            ox = int(round(math.cos(d) * 7))
            oy = int(round(math.sin(d) * 7))
            cv2.circle(res, (x,y), 3, color, 1, cv2.LINE_AA)
            cv2.line(res, (x,y), (x+ox,y-oy), color, 1, cv2.LINE_AA)        
    return res

fminutiae=draw_minutiae(fingerprint, minutiae)
cv2.imshow("fminutiae",fminutiae)
sminutiae=draw_minutiae(skeleton, minutiae)
cv2.imshow("sminutiae",sminutiae)




# A 1-pixel background border is added to the mask before computing the distance transform
mask_distance = cv2.distanceTransform(cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT), cv2.DIST_C, 3)[1:-1,1:-1]
cv2.imshow("mask_distance",mask_distance*255)


filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]]>10, minutiae))



fminutiae2=draw_minutiae(fingerprint, filtered_minutiae)
cv2.imshow("fminutiae2",fminutiae2)
sminutiae2=draw_minutiae(skeleton, filtered_minutiae)
cv2.imshow("sminutiae2",sminutiae2)
cv2.imwrite("traditionalThoeryImage/sminutiae2.bmp",sminutiae2)
def compute_next_ridge_following_directions(previous_direction, values):    
    next_positions = np.argwhere(values!=0).ravel().tolist()
    if len(next_positions) > 0 and previous_direction != 8:
        # There is a previous direction: return all the next directions, sorted according to the distance from it,
        # except the direction, if any, that corresponds to the previous position
        next_positions.sort(key = lambda d: 4 - abs(abs(d - previous_direction) - 4))
        if next_positions[-1] == (previous_direction + 4) % 8: # the direction of the previous position is the opposite one
            next_positions = next_positions[:-1] # removes it
    return next_positions


r2 = 2**0.5 # sqrt(2)

# The eight possible (x, y) offsets with each corresponding Euclidean distance
xy_steps = [(-1,-1,r2),( 0,-1,1),( 1,-1,r2),( 1, 0,1),( 1, 1,r2),( 0, 1,1),(-1, 1,r2),(-1, 0,1)]

# LUT: for each 8-neighborhood and each previous direction [0,8], 
#      where 8 means "none", provides the list of possible directions
nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]

def follow_ridge_and_compute_angle(x, y, d = 8):
    px, py = x, y
    length = 0.0
    while length < 20: # max length followed
        next_directions = nd_lut[cn_values[py,px]][d]
        if len(next_directions) == 0:
            break
        # Need to check ALL possible next directions
        if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
            break # another minutia found: we stop here
        # Only the first direction has to be followed
        d = next_directions[0]
        ox, oy, l = xy_steps[d]
        px += ox ; py += oy ; length += l
    # check if the minimum length for a valid direction has been reached
    return math.atan2(-py+y, px-x) if length >= 10 else None
valid_minutiae = []

def angle_abs_difference(a, b):
    return math.pi - abs(abs(a - b) - math.pi)

def angle_mean(a, b):
    return math.atan2((math.sin(a)+math.sin(b))/2, ((math.cos(a)+math.cos(b))/2))


for x, y, term in filtered_minutiae:
    d = None
    if term: # termination: simply follow and compute the direction        
        d = follow_ridge_and_compute_angle(x, y)
    else: # bifurcation: follow each of the three branches
        dirs = nd_lut[cn_values[y,x]][8] # 8 means: no previous direction
        if len(dirs)==3: # only if there are exactly three branches
            angles = [follow_ridge_and_compute_angle(x+xy_steps[d][0], y+xy_steps[d][1], d) for d in dirs]
            if all(a is not None for a in angles):
                a1, a2 = min(((angles[i], angles[(i+1)%3]) for i in range(3)), key=lambda t: angle_abs_difference(t[0], t[1]))
                d = angle_mean(a1, a2)                
    if d is not None:
        valid_minutiae.append( (x, y, term, d) )

dminutiae=draw_minutiae(fingerprint, valid_minutiae)
cv2.imshow("dminutiae",dminutiae)
cv2.imwrite("traditionalThoeryImage/dminutiae.bmp",dminutiae)




# Compute the cell coordinates of a generic local structure
# 计算
mcc_radius = 70
mcc_size = 16

g = 2 * mcc_radius / mcc_size
x = np.arange(mcc_size)*g - (mcc_size/2)*g + g/2
y = x[..., np.newaxis]
iy, ix = np.nonzero(x**2 + y**2 <= mcc_radius**2)
ref_cell_coords = np.column_stack((x[ix], x[iy]))


mcc_sigma_s = 7.0
mcc_tau_psi = 400.0
mcc_mu_psi = 1e-2

def Gs(t_sqr):
    """Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
    return np.exp(-0.5 * t_sqr / (mcc_sigma_s**2)) / (math.tau**0.5 * mcc_sigma_s)

def Psi(v):
    """Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
    return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))


# n: number of minutiae
# c: number of cells in a local structure

xyd = np.array([(x,y,d) for x,y,_,d in valid_minutiae]) # matrix with all minutiae coordinates and directions (n x 3)

# rot: n x 2 x 2 (rotation matrix for each minutia)
d_cos, d_sin = np.cos(xyd[:,2]).reshape((-1,1,1)), np.sin(xyd[:,2]).reshape((-1,1,1))
rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

# rot@ref_cell_coords.T : n x 2 x c
# xy : n x 2
xy = xyd[:,:2]
# cell_coords: n x c x 2 (cell coordinates for each local structure)
cell_coords = np.transpose(rot@ref_cell_coords.T + xy[:,:,np.newaxis],[0,2,1])

# cell_coords[:,:,np.newaxis,:]      :  n x c  x 1 x 2
# xy                                 : (1 x 1) x n x 2
# cell_coords[:,:,np.newaxis,:] - xy :  n x c  x n x 2
# dists: n x c x n (for each cell of each local structure, the distance from all minutiae)
dists = np.sum((cell_coords[:,:,np.newaxis,:] - xy)**2, -1)

# cs : n x c x n (the spatial contribution of each minutia to each cell of each local structure)
cs = Gs(dists)
diag_indices = np.arange(cs.shape[0])
cs[diag_indices,:,diag_indices] = 0 # remove the contribution of each minutia to its own cells

# local_structures : n x c (cell values for each local structure)
local_structures = Psi(np.sum(cs, -1))

def draw_minutiae_and_cylinder(fingerprint, origin_cell_coords, minutiae, values, i, show_cylinder = True):

    def _compute_actual_cylinder_coordinates(x, y, t, d):
        c, s = math.cos(d), math.sin(d)
        rot = np.array([[c, s],[-s, c]])    
        return (rot@origin_cell_coords.T + np.array([x,y])[:,np.newaxis]).T
    
    res = draw_minutiae(fingerprint, minutiae)    
    if show_cylinder:
        for v, (cx, cy) in zip(values[i], _compute_actual_cylinder_coordinates(*minutiae[i])):
            cv2.circle(res, (int(round(cx)), int(round(cy))), 3, (0,int(round(v*255)),0), 1, cv2.LINE_AA)
    return res
# @interact(i=(0,len(valid_minutiae)-1))
# def test(i=0):
'''
    for i in range(len(valid_minutiae)):
    cv2.imshow("xx"+str(i),draw_minutiae_and_cylinder(fingerprint, ref_cell_coords, valid_minutiae, local_structures, i))
'''


print(f"""Fingerprint image: {fingerprint.shape[1]}x{fingerprint.shape[0]} pixels
Minutiae: {len(valid_minutiae)}
Local structures: {local_structures.shape}""")


f1, m1, ls1 = fingerprint, valid_minutiae, local_structures
ofn = 'sample_1_2' # Fingerprint of the same finger
#ofn = 'samples/sample_2' # Fingerprint of a different finger
f2, (m2, ls2) = cv2.imread(f'{ofn}.png', cv2.IMREAD_GRAYSCALE), np.load(f'{ofn}.npz', allow_pickle=True).values()

# Compute all pairwise normalized Euclidean distances between local structures in v1 and v2
# ls1                       : n1 x  c
# ls1[:,np.newaxis,:]       : n1 x  1 x c
# ls2                       : (1 x) n2 x c
# ls1[:,np.newaxis,:] - ls2 : n1 x  n2 x c 
# dists                     : n1 x  n2
dists = np.sqrt(np.sum((ls1[:,np.newaxis,:] - ls2)**2, -1))
dists /= (np.sqrt(np.sum(ls1**2, 1))[:,np.newaxis] + np.sqrt(np.sum(ls2**2, 1))) # Normalize as in eq. (17) of MCC paper
# Select the num_p pairs with the smallest distances (LSS technique)
num_p = 5 # For simplicity: a fixed number of pairs
pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
score = 1 - np.mean(dists[pairs[0], pairs[1]]) # See eq. (23) in MCC paper
print(f'Comparison score: {score:.2f}') # 结果0.78
def draw_match_pairs(f1, m1, v1, f2, m2, v2, cells_coords, pairs, i, show_cylinders = True):
    #nd = _current_parameters.ND
    h1, w1 = f1.shape
    h2, w2 = f2.shape
    p1, p2 = pairs
    res = np.full((max(h1,h2), w1+w2, 3), 255, np.uint8)
    res[:h1,:w1] = draw_minutiae_and_cylinder(f1, cells_coords, m1, v1, p1[i], show_cylinders)
    res[:h2,w1:w1+w2] = draw_minutiae_and_cylinder(f2, cells_coords, m2, v2, p2[i], show_cylinders)
    for k, (i1, i2) in enumerate(zip(p1, p2)):
        (x1, y1, *_), (x2, y2, *_) = m1[i1], m2[i2]
        cv2.line(res, (int(x1), int(y1)), (w1+int(x2), int(y2)), (0,0,255) if k!=i else (0,255,255), 1, cv2.LINE_AA)
    return res
show_local_structures = False
'''
for i in range(len(pairs[0])):
    cv2.imshow("dfa"+str(i),draw_match_pairs(f1, m1, ls1, f2, m2, ls2, ref_cell_coords, pairs, i, show_local_structures))
'''



cv2.waitKey()
cv2.destroyAllWindows()
