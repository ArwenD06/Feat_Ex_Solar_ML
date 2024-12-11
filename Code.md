# Imports

We need all of the following imports:

```python

import numpy as np
import zarr
import matplotlib.pyplot as plt
import sunpy.map
import matplotlib.cm as cm
import dask.array as da
from sunpy.visualization import axis_labels_from_ctype, wcsaxes_compat
import cv2 as cv
import pandas as pd
import skimage as ski
import imageio.v3 as iio
import matplotlib.image as mpimg
from scipy import ndimage
import pyfeats as pf

from sklearn import feature_extraction
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from skimage import measure

from kneed import KneeLocator, DataGenerator

import matplotlib
import plotly.express as px

import statistics

```

# Saving and importing data
We saved our data using following commands.

```python
np.save("2011_05", selected_header)
np.save("2011_p05", selected_images)
```

We can open the used data by running the code below.
```python
headr = np.load('2015_01.npy', allow_pickle=True).item()
image = np.load('2015_p01.npy', allow_pickle=True)
```

Then you can make a sun map or plot the picture by running the following code.

```python
my_map = sunpy.map.Map((image, headr))

plt.figure(figsize=(4, 4))
ax = plt.subplot(projection=my_map)
my_map.plot()
```
```python
plt.figure(figsize=(4, 4))
plt.axis('off')
plt.imshow(image,origin='lower', vmin=0,vmax=1000, cmap='sdoaia193')
```

# Functions 

### different_masks(gray_image)
- Purpose: making different segmentations based on a grey scale image

- Input:
    * gray_image -> grey scale image you want to make segmentations of

- Output:
    * mask_otsu -> binary mask made by Otsu's method
    * mask_multi_otsu_l -> binary mask made by the first value of Multiple Otsu
    * mask_multi_otsu_h -> binary mask made by the last value of Multiple Otsu
    * mask_multi_otsu -> binary mask of the first and last value of Multiple Otsu combined
    * mask_yen -> binary mask made by Yen's method
    * mask_li -> binary mask made by Li's method
    * mask_min -> binary mask made by the minimum method


```python
def different_masks(gray_image):   
    t1 = ski.filters.threshold_otsu(gray_image)
    t2 = ski.filters.threshold_multiotsu(gray_image, 5)
    t3 = ski.filters.threshold_yen(gray_image)
    t4 = ski.filters.threshold_li(gray_image)
    t5 = ski.filters.threshold_minimum(gray_image)
    
    mask_otsu = gray_image > t1
    binary_mask2la = gray_image < t2[0]
    mask_multi_otsu_l = binary_mask2la*circle
    mask_multi_otsu_h = gray_image > t2[-1]
    mask_multi_otsu = mask_multi_otsu_l + mask_multi_otsu_h
    mask_yen = gray_image > t3
    mask_li = gray_image > t4
    binary_mask5a = gray_image < t5
    mask_min = binary_mask5a*circle
    
    return(mask_otsu, mask_multi_otsu_l, mask_multi_otsu_h, mask_multi_otsu, mask_yen, mask_li, mask_min)
```

### metrics(binary_mask1, binary_mask2l, binary_mask2h, binary_mask2, binary_mask3, binary_mask4, binary_mask5, binary_mask_a, binary_mask_b, binary_mask_total)
- Purpose: calculating the jaccard index and dice score for several segmentation methods
It's also possible to calculate recall, precision and the accuracy by uncommenting them

- Input:
    * binary_mask1, binary_mask2l, binary_mask2h, binary_mask2, 
                    binary_mask3, binary_mask4, binary_mask5 -> binary masks made by different segmentation techniques
    * binary_mask_a, binary_mask_b, binary_mask_total -> ground truths for active regions, coronal holes and both combined respectively
- Output:
    * all_dice -> a list with three sublist, the first sublist gives the dice scores for the comparison of binary_mask_a with the 7 input masks,
                 the second sublist gives the dice scores for the comparison of binary mask_b with the 7 input masks and
                  the last sublist gives the dice scores for th comparison of binary_mask_total with the 7 input masks.
    * all_jaccard -> a list with three sublists, the first sublist gives the jaccard indices for the comparison of binary_mask_a with the 7 input masks,
                 the second sublist gives the jaccard indices for the comparison of binary mask_b with the 7 input masks and
                  the last sublist gives the jaccard indices for th comparison of binary_mask_total with the 7 input masks.
```python
def metrics(binary_mask1, binary_mask2l, binary_mask2h, binary_mask2, 
                    binary_mask3, binary_mask4, binary_mask5, binary_mask_a, binary_mask_b, binary_mask_total):

    binary_masks = [binary_mask1, binary_mask2l, binary_mask2h, binary_mask2, 
                    binary_mask3, binary_mask4, binary_mask5]
    method = ["Otsu", "Multiple Otsu Lower without Background", "Multiple Otsu Higher", 
               "Multiple Otsu Total nb", "Yen", "Li", "Minimum without Background"]
    
    
    all_dice = []
    all_jaccard = []
    
    for mask in [binary_mask_a, binary_mask_b, binary_mask_total]:
        i=0
        j=0
        k=0
        FP=0
        FN=0
        TP=0
        TN=0

        dice = []
        jaccard = []
        binary_maskGT = mask
        while k < len(binary_masks):
            binary_mask_seg = binary_masks[k]
            FP=0
            FN=0
            TP=0
            TN=0
            i=0
            while i < len(binary_maskGT):
                j=0
                while j < len(binary_maskGT):
                    if binary_maskGT[i][j] == False and binary_mask_seg[i][j] == True:
                        FP = FP + 1
                    elif binary_maskGT[i][j] == True and binary_mask_seg[i][j] == False :
                        FN = FN + 1
                    elif binary_maskGT[i][j] == True and binary_mask_seg[i][j] == True:
                        TP = TP + 1
                    elif binary_maskGT[i][j] == False and binary_mask_seg[i][j] == False:
                        TN = TN + 1
                    j=j+1
                i = i+1

            #recall = TP / (TP + FN)
            #precision = TP / (TP + FP)
            f1 = 2*TP / (2*TP + FP + FN)
            iou = TP / (TP + FP + FN)
            #accuracy = TP / (TP + TN + FP + FN)
            dice.append(f1)
            jaccard.append(iou)
            k = k+1
        
        all_dice.append(dice)
        all_jaccard.append(jaccard)
        
    return(all_dice, all_jaccard)
```


### meaning(prediction, vector_final, sample_numbers)
- Purpose: splitting our data into the clusters

- Input: 
    * prediction -> list with numbers, prediction[i] = the cluster to which i belongs  
    * vector_final -> feature vector  
    * sample_numbers -> list with the sample numbers
    * k -> 

- Output:
    * klusters -> list with k sublists (k = number of klusters), every sublist j contains tuples for which the first element is the sample number of the 'blob', and the second element is the total number of the blob
    * feats -> list with k sublists, every sublist j contains the features of the 'blobs' that are part of kluster j

```python
def meaning(prediction, vector_final, sample_numbers, k):
    num_klus = max(prediction) +1
    nums = list(set(prediction))
    
    klusters = [ [] for _ in range(num_klus)]
    feats = [ [] for _ in range(num_klus)]
    
    j = k
    for el in prediction:
        for num in nums:
            if el == num:
                klusters[num].append((sample_numbers[j], j))
                feats[num].append(vector_final[j])
                j=j+1

    return klusters, feats
```


### slices_rect(tuples, sample_number, contours, used_slices)
- Purpose: finding the rectangles and contours that belong to a certain sample number i and cluster k

- Input:
    * tuples -> list with tuples from one cluster, example: one sublist obtained from the output klusters from the function meaning
    * sample_number -> sample sumber for which we want a visualisation 
    *  contours -> list with all the contours of all the 'blobs', can be obtained from the finding_contour function
    *  used_slices -> list of all the slices used to obtain the 'blobs'

- Output:
    * lx -> list with the length of the horizontal side of the rectangle
    * ly -> list with the length of the vertical side of the rectangle
    * rec -> list with tuples, each tuple (x, y) holds the x-coordinate of the bottom left corner of the rectangle and the y-coordinate of the bottom left corner of the rectangle
    * cons -> list with the contours that belong to the asked cluster and asked sample

```python
def slices_rect(tuples, sample_number, contours, used_slices):
    nr=[]
    rec = []
    lx=[]
    ly=[]
    cons = []
    for tup in tuples:
        if tup[0]== sample_number:
            cons.append(contours[tup[1]])
            nr.append(tup[1])
            slc = used_slices[tup[1]]
            y = slice.indices(slc[0],513)[0]
            ly.append(slice.indices(slc[0],513)[1] - slice.indices(slc[0],513)[0])
            x = slice.indices(slc[1],513)[0]
            lx.append(slice.indices(slc[1],513)[1] - slice.indices(slc[1],513)[0])
            rec.append((x,y))
            
    return lx, ly, rec, cons
```


### zero_border(arr)
- Purpose: creating a border of zeros around a 'blob'

- Input:
    * arr -> a numpy array representing the 'blob'
- Output:
    * c -> the 'blob' with a border of zeros around it

```python
def zero_border(arr):
    aa = np.insert(arr, 0, 0, axis=1)
    aaa = np.insert(aa, len(aa[0]), 0, axis=1)
    zero = [0 for i in range(len(aaa[0]))]
    b = np.insert(aaa, 0, zero, axis=0)
    c = np.append(b, [zero], axis=0)
    return c
```


### finding_contour(piece, sliced)
- Purpose: finding the contour of a certain 'blob'

- Input:
    * piece -> the 'blob' you want to find the contour of, represented as a numpy array
- Output:
    * contour -> the contour of the 'blob'

```python
def finding_contour(piece, sliced):
    zer_piec = zero_border(piece)
    contours = measure.find_contours(zer_piec, 0.8)
    if len(contours)>1:
        l = 0
        for contour in contours:
            if len(contour)>l:
                l=len(contour)
                contours[0] = contour

    y = slice.indices(sliced[0],513)[0]
    x = slice.indices(sliced[1],513)[0]
    for coor in contours[0]:
        coor[0] = coor[0] + y
        coor[1] = coor[1] + x
    return contours
```

# !!
For the following funtions to work properly, you need to run the following code. You need the file 'circle.png' for this.

```python
circle1 = cv.imread('circle.png', 0)
circle = circle1 > 0.5
```
### making_mask(gray_image)
- Purpose: making the binary mask of a sample, using Multiple Otsu's method. Binary_mask is the mask for the coronal holes and active regions combined, binary_maskl is the binary mask for the coronal holes only, and binary_maskh is the binary masj for active regions only, if you want to calculate the binary mask for the coronal holes or active regions seperatly, change the return statement to either binary_maskl or binary_maskh.

- Input:
    * gray_image -> the grey-scale image you want to make a mask of
- Output:
    * binary_mask -> the binary mask

```python
def making_mask(gray_image):
    t = ski.filters.threshold_multiotsu(gray_image, 5)

    binary_maskla = gray_image < t[0]
    binary_maskl = binary_maskla*circle
    binary_maskh = gray_image > t[-1]
    binary_mask = binary_maskl + binary_maskh
    
    return(binary_mask)
```


### make_bin_without_noise(binary_mask)
- Purpose: removing the noise from a binary mask

- Input:
    * binary_mask -> the binary mask you want to remove the noise of
- Output:
    * w_noise-> the binary mask without noise

```python
def make_bin_without_noise(binary_mask):
    mask_bin = np.zeros((len(binary_mask), len(binary_mask[0])))

    i=0
    while i < len(binary_mask):
        j=0
        while j < len(binary_mask[0]):
            if binary_mask[i][j] == True:
                mask_bin[i][j] = 1
            j=j+1
        i=i+1
    
    w_noise = ski.filters.median(mask_bin)
    return w_noise
```

### finding_slices(w_noise)
- Purpose: finding the slices to obtain every 'blob'

- Input:
    * w_noise -> the binary mask of your sample without noise
- Output:
    * slices -> a list with slices, every slice in this list gives you a 'blob'

 ```python
def finding_slices(w_noise):
    labeled_array, num_features = ndimage.label(w_noise)
    slices = ndimage.find_objects(labeled_array)
    return slices
```


### comps(sliced, gray_image, w_noise)
- Purpose: removing the noise from a binary mask

- Input:
    * sliced -> the slice that gives you the current 'blob'
    * gray_image -> the grey-scale images of the sample you're working with
    * w_noise -> the binary mask, without noise of the sample you're working with
- Output:
    * piece -> a binary mask of our 'blob', in a bounding box
    * selection -> the 'blob' in grey scale, with zeros on the pixels that don't belong to the 'blob', in a bounding box

```python
def comps(sliced, gray_image, w_noise):
    piece = w_noise[sliced]

    segmented = gray_image.copy()
    segmented[~binary_mask] = 0

    selection = segmented[sliced]
    return piece, selection
```

### roundness(area, perimeter)
- Purpose: calculation the roundness of a 'blob'  
roundness = $(4*\pi*area)/perimeter^2$

- Input:
    * area -> the area of the 'blob'
    * perimeter -> the perimeter of the 'blob'
- Output:
    * roundness -> the roundness of the 'blob'

```python
def roundness(area, perimeter):
    roundness = 4*np.pi*area/perimeter**2
    return roundness
```


### elongation(piece)
- Purpose: calculating the elongation of a 'blob'
enlongation = d_l/d_s, where d_l is the length of the longest side of the bounding box, 
                and d_s is the shortest side of the bounding box

- Input:
    * piece -> a binary mask of our 'blob', in a bounding box
- Output:
    * elongation -> the elongationg of the 'blob'

 ```python
def elongation(piece):
    if len(piece[0]) >= len(piece):
        d_l = len(piece[0])
        d_s = len(piece)
    else:
        d_s = len(piece[0])
        d_l = len(piece)

    elongation = d_l/d_s
    return elongation
```

### peri_on_area(area, perimeter)
- Purpose: calculation the ratio of the perimeter and area

- Input:
    * area -> the area of the 'blob'
    * perimeter -> the perimeter of the 'blob'
- Output:
    * ratio_peri_ar -> the ratio of the perimeter and area

 ```python
def peri_on_area(area, perimeter):
    ratio_peri_ar = perimeter / area
    return ratio_peri_ar
```


### features(selection, piece)
- Purpose: calculation of all the features for a 'blob'

- Input:
    * selection -> our 'blob' in grey scale, with zeros on the pixels that don't belong to the 'blob', in a bounding box  
    * piece -> a binary mask of our 'blob', in a bounding box
- Output:
    * feature -> a numpy array of length 27, with all the features   

```python
def features(selection, piece):
    piece_int = piece.astype(np.int32)
    features1, labels1 = pf.fos(selection, piece)
    features_mean, _, labels_mean, _ = pf.glcm_features(selection, ignore_zeros=True)
    features2 = ski.measure.regionprops(piece_int, selection)
    
    feature = np.array([])
    feature = np.append(feature,features1[0:2])
    feature = np.append(feature,features1[6:10])
    feature = np.append(feature,features_mean[:-1])
    feature = np.append(feature,features2[0]['area'])
    feature = np.append(feature,features2[0]['perimeter'])
    feature = np.append(feature,features2[0]['eccentricity'])
    feature = np.append(feature,features2[0]['extent'])
    feature = np.append(feature,features2[0]['solidity'])
    feature = np.append(feature,roundness(features2[0]['area'], features2[0]['perimeter']))
    feature = np.append(feature,elongation(piece))
    feature = np.append(feature,peri_on_area(features2[0]['area'], features2[0]['perimeter']))
    return feature
```

# Calculating the feature vector

Choose how many samples a year you want.
```python
a = 3
```

Making a list with the names of the samples you want to use.
```python
images = []
headers = []
i=0

while i < 10:
    j=1
    while j < a:
        headr = '201' + str(i) + '_0' + str(j) + '.npy'
        image = '201' + str(i) + '_p0' + str(j) + '.npy'
        headers.append(headr)
        images.append(image)
        j=j+1
    i=i+1
```

List of the names of all of the used features.
```python
names = ['Mean', 'Variance', 'Energy', 'Entropy', 'Minimal Grey Level', 'Maximal Grey Level', 'ASM', 'Contrast',
         'Correlation', 'Sum Of Squares Variance', 'Inverse Difference Moment', 'Sum Average', 'Sum Variance', 'Sum Entropy',
         'Entropy', 'Difference Variance', 'Difference Entropy', 'Randomness', 'Dependency', 'Area', 'Perimeter',
         'Eccentricity', 'Extent', 'Solidity', 'Roundness', 'Elongation', 'Ratio perimeter and area']
```

Calculating the feature vector.
```python
masks = []
all_features = []
sample_number = []
used_slices=[]
contours = []

i=0

for date in images:
    image = np.load(date, allow_pickle=True)
    
    plt.imsave('img.png', image, origin='lower', vmin=0,vmax=1000, cmap='sdoaia193', dpi=100)
    gray_image = cv.imread('img.png', 0)
    
    binary_mask = making_mask(gray_image)
    bin_w_noise = make_bin_without_noise(binary_mask)
    masks.append(bin_w_noise)
    slices = finding_slices(bin_w_noise)
    i=i+1
    for sliced in slices:
        piece, selection = comps(sliced, gray_image, bin_w_noise)
        if len(piece)<4:
            continue
        if len(piece[0])<4:
            continue
        try:
            feature = features(selection, piece)
            contour = finding_contour(piece, sliced)
            contours.append(contour)
            all_features.append(feature)
            sample_number.append(i)
            used_slices.append(sliced)
        except np.linalg.LinAlgError:
            pass
    print('Sample ' + str(i-1) + ' done')  
```

# Histogram of a single sample

Choose the sample you want a histogram of.
```python
image = np.load('2012_p01.npy', allow_pickle=True)
```

```python
plt.figure(figsize=(5,5))
plt.axis('off')
plt.imshow(image, vmin=0,vmax=1000, cmap='sdoaia193')
plt.savefig('img.jpg', bbox_inches='tight', pad_inches=0)

sun = iio.imread('img.jpg')
gray_image = ski.color.rgb2gray(sun)

histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0.0, 1.0))
fig, ax = plt.subplots()
ax.plot(bin_edges[0:-1], histogram)
ax.set_title("Graylevel histogram")
ax.set_xlabel("gray value")
ax.set_ylabel("pixel count")
ax.set_xlim(0, 1.0)

```
# Trying different segmentation techniques
```python
ski.filters.try_all_threshold(gray_image, figsize=(8, 5), verbose=True)
```

# Segmentation Evaluation

Can be combined with the making of the feature vector, but the following code takes a while to run, so if it's not needed, you can skip the next two cells.
```python
imgs = []
total = []
bina = []
binb = []
i=0
j=1
while i < 10:
    tot = '201' + str(i) + '_total' + str(j) + '.png'
    bin1a = '201' + str(i) + '_bin' + str(j) + 'a.png'
    bin1b = '201' + str(i) + '_bin' + str(j) + 'b.png'
    image = '201' + str(i) + '_p0' + str(j) + '.npy'
    total.append(tot)
    bina.append(bin1a)
    binb.append(bin1b)
    imgs.append(image)
    i=i+1
```

```python
bin_high_dice = []
bin_low_dice = []
bin_tot_dice = []
bin_high_jacc = []
bin_low_jacc = []
bin_tot_jacc = []

i=0

for date in imgs:
    image = np.load(date, allow_pickle=True)

    plt.imsave('img.png', image, origin='lower', vmin=0,vmax=1000, cmap='sdoaia193', dpi=100)
    gray_image = cv.imread('img.png', 0)

    mask_total = cv.imread(total[i], 0)
    binary_mask_total = mask_total > 0.5

    mask_a = cv.imread(bina[i], 0)
    binary_mask_a = mask_a > 0.5

    mask_b = cv.imread(binb[i], 0)
    binary_mask_b = mask_b > 0.5

    mask_otsu, mask_multi_otsu_l, mask_multi_otsu_h, mask_multi_otsu, mask_yen, mask_li, mask_min = different_masks(gray_image)

    all_dice, all_jaccard = metrics(mask_otsu, mask_multi_otsu_l, mask_multi_otsu_h, mask_multi_otsu, 
                                    mask_yen, mask_li, mask_min, binary_mask_a, binary_mask_b, binary_mask_total)

    bin_high_dice.append(all_dice[0]) 
    bin_low_dice.append(all_dice[1])
    bin_tot_dice.append(all_dice[2]) 

    bin_high_jacc.append(all_jaccard[0]) 
    bin_low_jacc.append(all_jaccard[1]) 
    bin_tot_jacc.append(all_jaccard[2])
    
    i=i+1
```

```python
scores = {'Otsu': [], 'Multiple Otsu Higher': [], 'Yen':[], 'Li':[],
        'Minimum without Background': [], 'Multiple Otsu Lower without Background': [], 'Multiple Otsu Total nb': [] }

methods = ["Otsu", "Multiple Otsu Lower without Background", "Multiple Otsu Higher", 
               "Multiple Otsu Total nb", "Yen", "Li", "Minimum without Background"]
```

```python
rn = len(total)

for key in scores:
    index = methods.index(key)
    if key in ['Otsu', 'Multiple Otsu Higher', 'Yen', 'Li']:
        lst_d = [bin_high_dice[i][index] for i in range(0,rn)]
        lst_j = [bin_high_jacc[i][index] for i in range(0,rn)]
        scores[key] = [round(statistics.mean(lst_d), 6), round(statistics.mean(lst_j), 6)]
    elif key in ['Minimum without Background', 'Multiple Otsu Lower without Background']:
        lst_d = [bin_low_dice[i][index] for i in range(0,rn)]
        lst_j = [bin_low_jacc[i][index] for i in range(0,rn)]
        scores[key] = [round(statistics.mean(lst_d), 6), round(statistics.mean(lst_j), 6)]
    else:
        lst_d = [bin_tot_dice[i][index] for i in range(0,rn)]
        lst_j = [bin_tot_jacc[i][index] for i in range(0,rn)]
        scores[key] = [round(statistics.mean(lst_d), 6), round(statistics.mean(lst_j), 6)]
```

# Making a dataframe of the feature vector
```python
X = pd.DataFrame(all_features)
```

# Removing features & standardizing
Standardising the data: 
```python
feats_all = all_features.copy()
stan = StandardScaler().fit(feats_all)
feats_stan_all = stan.transform(feats_all)
```
Removing correlated features of original feature vector:

```python
copy1_vector_final = all_features.copy()
vector_non_stan = np.delete(copy1_vector_final, [6,11,13,14,16], 1)
```
Removing correlated features from standardized feature vector.
```python
copy1_feats_stan_all = feats_stan_all.copy()
feats_final = np.delete(copy1_feats_stan_all, [6,11,13,14,16], 1)
```

Removing the names of the features that we don't use anymore.
```python
names_1 = names.copy()
names_final = np.delete(names_1, [6,11,13,14,16], 0)
```

# Correlation Matrix
```python
matrix_all = np.transpose(feats_stan_all)
corr_matrix_all = np.corrcoef(matrix_all)

matrix_final = np.transpose(feats_final)
corr_matrix_final = np.corrcoef(matrix_final)
```


```python
ticks_all = tuple(range(0,len(matrix_all)))
labels_all = tuple(range(1,len(matrix_all)+1))

ticks_final = tuple(range(0,len(matrix_final)))
labels_final = tuple(range(1,len(matrix_final)+1))

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14, 6))
im1 = ax1.imshow(corr_matrix_all)
im1.set_clim(-1, 1)
ax1.grid(False)
ax1.xaxis.set(ticks=ticks_all, ticklabels=labels_all)
ax1.yaxis.set(ticks=ticks_all, ticklabels=labels_all)

im2 = ax2.imshow(corr_matrix_final)
im2.set_clim(-1, 1)
ax2.grid(False)
ax2.xaxis.set(ticks=ticks_final, ticklabels=labels_final)
ax2.yaxis.set(ticks=ticks_final, ticklabels=labels_final)

cbar1 = ax1.figure.colorbar(im1, ax=ax1, format='% .2f', shrink=.85, pad=.05, aspect=8)
cbar2 = ax2.figure.colorbar(im2, ax=ax2, format='% .2f', shrink=.85, pad=.05, aspect=8)

plt.show()
```

# t-SNE plots

```python
X_c = feats_final.copy()

tsne = TSNE(n_components=2, random_state=100)

X_tsne = tsne.fit_transform(X_c)
```

```python
Y = sample_number

plt.figure(figsize=(8, 6))
scatter2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis')

plt.legend(*scatter2.legend_elements(num=19), title="Sample Number",bbox_to_anchor=(1.25, 1), loc='upper right')
plt.title("t-SNE Visualization of Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
```

# PCA plots

```python
X_comp1 = feats_final.copy()

pca = PCA(n_components=2, random_state=56)

X_pca = pca.fit_transform(X_comp1)
```

```python
Y = sample_number

plt.figure(figsize=(8, 6))
scatter2 = plt.scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=Y)

plt.legend(*scatter2.legend_elements(num=19), title="Sample Number", bbox_to_anchor=(1.25, 1), loc='upper right')
plt.title("PCA Visualization of Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
```

# K-means

Choose number of clusters n and how many training data you want k (k = where you want the training data to stop, thus k-1 is the last element of your training data).
```python
n = 5
k = 100
```

## t-SNE
```python
trainData = X_tsne[:k]
kmeans = KMeans(n_clusters=n, random_state=3).fit(trainData)
prediction_tsne = kmeans.predict(X_tsne[k:])

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[k:, 0], X_tsne[k:, 1], c=prediction_tsne, cmap='gist_rainbow')
#scatter2 = plt.scatter(X_tsne[:k, 0], X_tsne[:k, 1], c=kmeans.labels_, cmap='viridis')


plt.legend(*scatter.legend_elements(), title="Label")
plt.title("Clusters formed based on t-SNE components using K-means clustering")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
```

## PCA
```python
trainData_pca = X_pca[:k]
kmeans = KMeans(n_clusters=n, random_state=57).fit(trainData_pca)
prediction_pca = kmeans.predict(X_pca[k:])

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[k:, 0], X_pca[k:, 1], c=prediction_pca, cmap='gist_rainbow')
#scatter2 = plt.scatter(X_pca[:k, 0], X_pca[:k, 1], c=kmeans.labels_, cmap='viridis')


plt.legend(*scatter.legend_elements(), title="Label")
plt.title("Clusters formed based on PCA components using K-means clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
```

# Meaning of the clusters
Choose for which data you want the visualizations, put the other one in a comment.
```python
prediction = prediction_tsne
prediction = prediction_pca
```

## Histograms
```python
_, feats = meaning(prediction, feats_final, sample_number, k)
alp = [1, 0.85, 0.65, 0.45, 0.25, 0.1]

plt.figure(figsize=(10, 15)) 
for i in range(0, 22):
    plt.subplot(8, 3, i+1)
    for j in range(0,len(feats)):
        feats_df = pd.DataFrame(feats[j])
        plt.hist(feats_df[i], bins=25, alpha=alp[j])
    plt.title(names_final[i])
    plt.xlabel('Feature range')
    plt.ylabel("Amount")  
    plt.tight_layout()
```

## Numerical
```python
_, feats = meaning(prediction, feats_final, sample_number, k)

means = []
ranges = []

i=0
while i < len(feats):
    m = []
    r = []
    feats_df = pd.DataFrame(feats[i])
    for j in range(0,22) :
        column = feats_df[j]
        m.append(round(np.mean(column), 4))
        r.append(round((np.max(column)-np.min(column)), 4))
    means.append(m)
    ranges.append(r)
    i=i+1
```
Putting the means and the ranges of all of the features in a dataframe to make them visually clear.
```python
d_means = pd.DataFrame(np.transpose(means))
d_ranges = pd.DataFrame(np.transpose(ranges))
```

## Visual of a sample
```python
image1 = np.load('2012_p01.npy', allow_pickle=True)
plt.imsave('img1.png', image1, origin='lower', vmin=0,vmax=1000, cmap='sdoaia193', dpi=100)

sun = iio.imread('img1.png')

cmap = cm.get_cmap('gist_rainbow', n)

klusters, _  = meaning(prediction, vector_non_stan, sample_number, k)

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(sun, vmin=0,vmax=1000, cmap='sdoaia193')
ax2.imshow(sun, vmin=0,vmax=1000, cmap='sdoaia193')

ax1.axis('off')
ax2.axis('off')

plt.tight_layout()

for j in range(0, n):
    lx, ly, rec, cons = slices_rect(klusters[j], 5, contours, used_slices)
    for contour in cons:
        ax2.plot(contour[0][:, 1], contour[0][:, 0], linewidth=2, c=cmap(j))
```

## Applying the Kneedle method
Choose for which data you want to determine the Kneedle point, for the PCA or the t-SNE, uncomment that one and put the other one in a comment.

```python
data = X_pca
#data = X_tsne
```

```python
sse = []
list_k = list(range(2, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(data)
    sse.append(km.inertia_)

kneedle = KneeLocator(list_k, sse, S=1.0, curve='convex', direction='decreasing')
print("Kneelde point is located at " + str(kneedle.elbow))

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.title("Kneedle method")
plt.plot(list_k, sse, '-o')
plt.axvline(kneedle.elbow, ls='--', c='0.5')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distance');
```

# 3D plots

For the 3D plots you need to run the following line first.
```python
get_ipython().run_line_magic('matplotlib', 'widget')
```

## 3D plot PCA
```python
X_c3 = feats_final.copy()

pca = PCA(n_components=3)

X_pca_3d = pca.fit_transform(X_c3)
```
```python
Y = sample_number

fig_pca = px.scatter_3d(X_pca_3d,
                    x=X_pca_3d[:,0],
                    y=X_pca_3d[:,1],
                    z=X_pca_3d[:,2],
                    title="PCA Visualization of Dataset",
                    opacity=1,
                    color=Y,
                    size= [1]*len(X_pca_3d),
                    labels={'x':'PCA component 1','y':'PCA component 2','z':'PCA component 3'},
                    color_continuous_scale='viridis'
                    )

fig_pca.show()
```
If you want to export the plot to html, run the next line.
```python
fig_pca.write_html('PCA_final.html')
```

## 3D plot t-SNE
```python
X_c_3 = feats_final.copy()

tsne = TSNE(n_components=3, random_state=42) 

X_tsne_3d = tsne.fit_transform(X_c_3)
```

```python
Y = sample_number

fig_tsne = px.scatter_3d(X_tsne_3d,
                    x=X_tsne_3d[:,0],
                    y=X_tsne_3d[:,1],
                    z=X_tsne_3d[:,2],
                    title="t-SNE Visualization of Dataset",
                    opacity=1,
                    color=Y,
                    size= [1]*len(X_tsne_3d),
                    labels={'x':'t-SNE component 1','y':'t-SNE component 2','z':'t-SNE component 3'},
                    color_continuous_scale='viridis'
                    )
fig_tsne.show()
```

If you want to export the plot to html, run the next line.
```
fig_tsne.write_html('tsne_final.html')
```

## 3D plot correlation
```python
vector = feats_stan_all

Y = sample_number

fig_corr = px.scatter_3d(vector,
                    x=vector[:,3],
                    y=vector[:,13],
                    z=vector[:,14],
                    title="Correlation",
                    opacity=1,
                    color=Y,
                    size=[5]*len(vector),
                    labels={'x':names_final[3],'y':names_final[13],'z':names_final[14]},
                    color_continuous_scale='viridis'
                    )

fig_corr.show()
```

If you want to export the plot to html, run the next line.

```python
fig_corr.write_html('corr.html')
```
