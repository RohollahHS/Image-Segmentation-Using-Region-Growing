import numpy as np
import cv2
import matplotlib.pyplot as plt

def display(img, vmin=0, vmax=255):
    plt.figure(figsize=(16, 9), tight_layout=True)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()

# 3 --------------------------------------
# 3-1 ------------------------------------

mri = cv2.imread('images/Color_MRI.png', 0)

# 3-2 ------------------------------------

white_seed_x, white_seed_y = 350, 390
white_seed = mri[white_seed_x, white_seed_y]

gray_seed_x, gray_seed_y = 290, 280
gray_seed = mri[gray_seed_x, gray_seed_y]

# 3-3 ------------------------------------

mri_seed = np.zeros_like(mri)
mri_seed[white_seed_x, white_seed_y] = white_seed
mri_seed[gray_seed_x, gray_seed_y] = gray_seed

# 3-4 ------------------------------------

def dilation(img, threshold):
    dilated = np.zeros(shape=(img.shape[0]+2, img.shape[1]+2), dtype=np.int16)
    m, n = np.where(img == threshold)
    for x, y in zip(m, n):
        pixels = np.array([[x-1, y  ],
                           [x+1, y  ],
                           [x  , y-1],
                           [x  , y+1],
                           [x  , y  ]])
        dilated[pixels[:, 0], pixels[:, 1]] = threshold
    return dilated[0:-2, 0:-2].astype(np.int16)


def region_growing(img, growing_img, threshold, T, threshold_type):
    dilated = dilation(growing_img, threshold)
    neighbours = np.zeros_like(dilated)
    dilated[growing_img==threshold] = 0
    neighbours = dilated.copy()

    if threshold_type == 'constant':
        m, n = np.where(neighbours == threshold)
        for i, j in zip(m, n):
            if abs((threshold - img[i, j])) <= T:
                growing_img[i, j] = threshold

    else: 
        k, l = np.where(growing_img == threshold)
        new_threshold = round(np.mean(img[k, l]))
        m, n = np.where(neighbours == threshold)
        for i, j in zip(m, n):
            if abs((new_threshold - img[i, j])) <= T:
                growing_img[i, j] = threshold

    return growing_img

# 3-5,6,7 ------------------------------------

# iteration = 10000
# mri = np.array(mri, dtype=np.int16)
# growing_img = mri_seed.copy()
# T = [10, 15, 20, 25, 30, 35, 40, 45, 50]
# white_segments = []
# for t in T:
#     growing_img = mri_seed.copy()
#     for i in range(iteration):
#         x = np.sum(growing_img)
#         growing_img = region_growing(img=mri, growing_img=growing_img, 
                                       #threshold=white_seed, T=t, threshold_type='constant')
#         if np.sum(growing_img) == x:
#             print('Number of Iteration: ', i+1)
#             white_segments.append(growing_img)
#             break



# fig = plt.figure(figsize=(15, 15))
# # plt.title('Erosion iterations')
# plt.axis('off')
# for i in range(len(T)):
#     fig.add_subplot(3, 3, i+1)
#     plt.imshow(white_segments[i], cmap='gray')
#     plt.title(f'T={T[i]}')
#     plt.axis('off')
# plt.show()


iteration = 10000
mri = np.array(mri, dtype=np.int16)
growing_img = mri_seed.copy()
for i in range(iteration):
    x = np.sum(growing_img)
    growing_img = region_growing(img=mri, growing_img=growing_img, threshold=white_seed, 
                                 T=35, threshold_type='constant')
    if np.sum(growing_img) == x:
        print('Number of Iteration: ', i+1)
        break
    
white_segment = growing_img.copy()
# display(white_segment, 0, np.max(white_segment))


# iteration = 10000
# mri = np.array(mri, dtype=np.int16)
# growing_img = mri_seed.copy()
# T = [10, 15, 20, 25, 30, 35, 40, 45, 50]
# gray_segments = []
# for t in T:
#     growing_img = mri_seed.copy()
#     for i in range(iteration):
#         x = np.sum(growing_img)
#         growing_img = region_growing(img=mri, growing_img=growing_img, threshold=gray_seed, 
                                       #T=t, threshold_type='constant')
#         if np.sum(growing_img) == x:
#             print('Number of Iteration: ', i+1)
#             gray_segments.append(growing_img)
#             break


# fig = plt.figure(figsize=(15, 15))
# # plt.title('Erosion iterations')
# plt.axis('off')
# for i in range(len(T)):
#     fig.add_subplot(3, 3, i+1)
#     plt.imshow(gray_segments[i], cmap='gray')
#     plt.title(f'T={T[i]}')
#     plt.axis('off')
# plt.show()


iteration = 10000
mri = np.array(mri, dtype=np.int16)
growing_img = mri_seed.copy()
for i in range(iteration):
    x = np.sum(growing_img)
    growing_img = region_growing(img=mri, growing_img=growing_img, threshold=gray_seed, 
                                 T=20, threshold_type='constant')
    if np.sum(growing_img) == x:
        print('Number of Iteration: ', i+1)
        break
    
gray_segment = growing_img.copy()
# display(gray_segment, 0, np.max(gray_segment))


iteration = 10000
mri = np.array(mri, dtype=np.int16)
growing_img = mri_seed.copy()

for i in range(iteration):
    growing_img_ = growing_img.copy()
    growing_img = region_growing(img=mri, growing_img=growing_img, threshold=white_seed, 
                                 T=35, threshold_type='variable')
    if np.sum(growing_img) == np.sum(growing_img_):
        print('Number of Iteration: ', i+1)
        break

white_segment_var = growing_img.copy()
# display(white_segment_var, 0, np.max(white_segment_var))


iteration = 10000
growing_img = mri_seed.copy()
for i in range(iteration):
    growing_img_ = growing_img.copy()
    growing_img = region_growing(img=mri, growing_img=growing_img, threshold=gray_seed, 
                                 T=20, threshold_type='variable')
    if np.sum(growing_img) == np.sum(growing_img_):
        print('Number of Iteration: ', i+1)
        break
gray_segment_var = growing_img.copy()
# display(gray_segment_var, 0, np.max(gray_segment_var))


import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(19, 12), tight_layout=True)
gs = gridspec.GridSpec(2, 3)
row = 2
col = 3
fig.add_subplot(gs[:, 0])
plt.imshow(mri, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image', fontsize=14)
plt.axis('off')
fig.add_subplot(row, col, 2)
plt.imshow(white_segment, cmap='gray')
plt.title('white segment constant', fontsize=14)
plt.axis('off')
fig.add_subplot(row, col, 3)
plt.imshow(gray_segment, cmap='gray')
plt.title('gray segment constant', fontsize=14)
plt.axis('off')
fig.add_subplot(row, col, 5)
plt.imshow(white_segment_var, cmap='gray')
plt.title('white segment variable', fontsize=14)
plt.axis('off')
fig.add_subplot(row, col, 6)
plt.imshow(gray_segment_var, cmap='gray')
plt.title('gray segment variable', fontsize=14)
plt.axis('off')
plt.show()

# 4 --------------------------------------
# 4-1 ------------------------------------
 
mri_refrence = cv2.imread('images/Color_MRI.png', 0)
mri_distorted = cv2.imread('images/Color_MRI2.png', 0)

# 4-2 ------------------------------------

def user_point():

    REFRENCE = mri_refrence.copy()
    DISTORTED = mri_distorted.copy()
    global points_ref
    global points_dist
    points_ref = []
    points_dist = []

    def enter_points(event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            cv2.circle(REFRENCE, (x, y), 4, (255, 0, 0), -1)
            points_ref.append([x, y])

    cv2.namedWindow('REFRENCE')
    cv2.setMouseCallback('REFRENCE', enter_points)

    def enter_points(event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            cv2.circle(DISTORTED, (x, y), 4, (255, 0, 0), -1)
            points_dist.append([x, y])
            
    cv2.namedWindow('DISTORTED')
    cv2.setMouseCallback('DISTORTED', enter_points)

    while(1):
        cv2.imshow('REFRENCE', REFRENCE)
        cv2.imshow('DISTORTED', DISTORTED)
        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    return np.array(points_ref), np.array(points_dist)


ps_ref, ps_dist = user_point()


# # prefred 11 points
# ps_ref = np.array([[215, 247],
#                    [244, 184],
#                    [395, 153],
#                    [417, 155],
#                    [559, 179],
#                    [611, 244],
#                    [580, 510],
#                    [458, 667],
#                    [380, 669],
#                    [233, 546],
#                    [236, 430]])

# ps_dist = np.array([[140, 347],
#                     [115, 270],
#                     [194, 149],
#                     [209, 140],
#                     [321,  78],
#                     [407, 110],
#                     [578, 376],
#                     [611, 599],
#                     [561, 648],
#                     [371, 615],
#                     [289, 505]])


# 4-3 ------------------------------------

h, _ = cv2.findHomography(ps_dist, ps_ref, cv2.RANSAC)
 
height, width = mri_distorted.shape
mri_registered = cv2.warpPerspective(mri_distorted, h, (width, height))


def joint_histogram(img_refrence, img_registered, title, dasteh=20):

    h = cv2.calcHist([img_refrence, img_registered], [0, 1], None, 
                     [dasteh, dasteh], [0, 256, 0, 256])

    from mpl_toolkits import mplot3d
    import matplotlib.cm as cm

    # setup the figure and axes
    fig = plt.figure(figsize=(20, 15))
    plt.axis('off')
    ax1 = fig.add_subplot(111, projection='3d')

    _x = np.arange(0, dasteh, 1)
    _y = np.arange(0, dasteh, 1)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    dz = h.ravel()
    bottom = np.zeros_like(dz)
    width = depth = 1
    cmap = cm.get_cmap('Greys') # Get desired colormap - you can change this!
    max_height = np.max(dz) # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz]
    ax1.bar3d(x, y, bottom, width, depth, dz, color=rgba, zsort='average')
    ax1.set_title(title)
    ax1.set_xticks(np.arange(0, dasteh), 
                   np.arange(0, 256, round(255/dasteh)), fontsize=8)
    ax1.set_yticks(np.arange(0, dasteh), 
                   np.arange(0, 256, round(255/dasteh)), fontsize=8)
    plt.show()


joint_histogram(mri_refrence, mri_distorted, 'mri_refrence and mri_distorted', 20)
joint_histogram(mri_refrence, mri_registered, 'mri_refrence and mri_registered', 20)
joint_histogram(mri_refrence, mri_refrence, 'mri_refrence and mri_refrence', 20)


