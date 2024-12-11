# Feature Extraction From Full Disk Solar Images Using Morphological Filters and Machine Learning
## *Abstract*
*The Sun’s activity drives various dangerous phenomena, like solar flares and CME’s. These events can cause serious troubles, as they have caused communication problems and power grid failures in the past. Understanding these events is both useful and necessary. This project aims to analyse solar images using simple techniques and light machine learning algorithms. Providing an alternative to computationally expensive frameworks, like deep learning and Convolutional Neural Networks (CNN). We used image processing methods, including thresholding techniques like Otsu’s method, to segment solar images. Key features, such as intensity and area, were extracted to create a feature vector. The dimensions of our feature vector were reduced using PCA and t-SNE. Then we clustered, making use of K-means clustering. Otsu’s method was found to be the best segmentation method for our data. K-means and t-SNE effectively found clusters , describing different regions. Our project shows that there is potential in using these more straightforward techniques in solar data analysis.*

## Summary
The activity of the Sun causes many dangerous phenomena, such as solar flares, storms and winds. They can cause many problems for our satellites and power grids. For this reason, it is important to study and understand these phenomena. They are caused by the complex magnetic field of the Sun, it becomes tangled and contains a massive amount of energy. When these knots burst, the energy is released in to space.
The Solar Dynamics Observatory (SDO) is a satellite in orbit around the Sun. Since its launch on the 11th of February 2010, it has been providing us with continuous data of our Sun in multiple formats. The satellite contains multiple instruments, but we will be focusing on the Atmospheric Imaging Assembly (AIA). This instrument contains four telescopes, from which images are made of two ultraviolet bands, seven extreme ultraviolet bands and one visible wavelength. This is the data we will be using for our project.
We have processed images, starting by creating histograms. After this, we worked with automatic thresholding methods, of which the Otsu method was the superior option. We determined this by applying evaluation metrics to find the best thresholding method. After thresholding, we can apply morphological filters, such as dilation and erosion. These operations will fill holes and separate objects.
After the segmentation, we can study our separated regions. We do this using feature extraction, a technique where you calculate many different features. These describe things like shape, area and intensity. We have defined many different features, combined in a feature vector. Using this vector, we can start with machine learning.
Before we put these function vectors in algorithms, we have to carry through a few more changes. We start by standardising all features and computing their correlation. Features that are correlated can be reduced into one feature. After this, we used PCA and t-SNE to reduce the dimension of our features. Lastly, we used K-means clustering to group our segmented regions based on the previous calculated features, each cluster will represent a different kind of region of our Sun.

## State of the Art
In our bachelor project we set out to study the sun, using simple operations and light machine learning techniques. There are already very successful frameworks that can achieve this, but they use heavy deep learning algorithms, like Convolutional Neural Networks (CNN). In this research they developed SCSS-net [16]. The U-net deep learning architecture has also been applied in solar physics, but they have encountered limitations because of GPU requirements [34]. Though effective, these algorithms require significant computational resources and time. We aim to get similar results with simpler, yet efficient methods.

## The Data
Although raw data from the SDO is available, it comes with a few risks. Especially when wanting to perform machine learning, one first needs to modify this raw data to a more usable form. Corrupt observations or the degradation of instruments over time will cause trouble when training a model. For these reason, we will use a curated data set from Galvez et al [8]. This data set handles a lot of the issues of working with raw data, which permits data scientists to apply machine learning without needing in-depth heliophysical knowledge. We will focus on the problems solved for AIA data, since we will be using this data.
For one, instrument degradation is corrected. When instruments such as those on the SDO produce data for years on end, there is bound to be degradation over time, mostly due to CCD corrosion. Since AIA is calibrated against EVE, which in turn is in connection with external measurements, one can determine the time dependent AIA data distribution independently of solar cycles. This makes it possible to counteract the degradation over time by realigning our data with the predicted result. Secondly, corrupt observations are removed. This can occur for a number of reasons, such as data taken during instrument calibration, data taken when other objects are between the SDO and the Sun, or they may just be random instrumental anomalies. These images can be isolated by looking at abnormal jumps or non-physical behaviour in the average AIA wavelength band data counts. Note that the degradation of instruments will also be visible under a monotone downward trend. Lastly, due to the elliptical orbit of the SDO around the Sun, the Sun visibly varies in size throughout the year. To solve this, all images are scaled to align with a constant solar radius. While these data corrections are being applied, the images are sized down to a lower resolution of 512 × 512, by summing and averaging local blocks. This will make the data more practical for machine learning purposes. Additionally, the AIA, HMI and EVE data are synchronised time wise, where AIA and EVE observations have a 6-minute cadence, while the HMI has a 12-minute cadence. So although the times are matched, HMI will occur every other step compared to AIA and EVE data.
The final curated data set contains roughly 6.5 T B of data, ordened in time and all in NumPy format. Galvez et al [8] furthermore provides us with data splits to ensure proper use in machine learning. Random data splits might result in training and testing on observation only days, hours or minutes apart. In the context of solar activities, these data points will have almost identical structures, resulting in a worse learning process.
The dataset of Galvez et al [8] was originally published in Numpy’s compressed array format (.npz). For this project we will instead be using the updated SDOML dataset, in which all data is converted to the Zarr format (.zarr). This dataset is available on the SDOML website [12]. The Zarr format is an upgrade to the NumPy array and is used for storing N-dimensional arrays or tensors. The biggest advantage of this new dataset is the ability to use data from the cloud, without the need of downloading it ourself. This can be done using the Zarr-python package [32], which is specially designed to handle files with the Zarr format.

## Thresholding

Using the scikit-image package, we can apply all the thresholding algorithms as previously described. In particular, we use the try all threshold function [22] to get an overview of results of multiple thresholding methods. This will help to decide which method is best suited for our research. For working with one specific thresholding method, scikit-image provides specific thresholding function, such as the threshold multiotsu function [22] for the Multiple Otsu method. When applying Multiple Otsu, we asked for five classes, which then resulted in four thresholds. For the segmentations, we then used the first and last threshold.
<img width="725" alt="Scherm­afbeelding 2024-12-10 om 21 58 36" src="https://github.com/user-attachments/assets/1f2fba98-7a25-46bf-9f7a-07274d5fc09b">
<img width="728" alt="Scherm­afbeelding 2024-12-10 om 21 59 30" src="https://github.com/user-attachments/assets/a2935efb-0c77-4005-b597-c398110473bf">

For the first four rows of the tables 3.1 and 3.2 we compared the segmentations made by Otsu’s method, Multiple Otsu’s method (the first obtained value), Yens method and Li’s method with our ground truth for the active regions. In row five and six, we compared the Minimum method and the Multiple Otsu method (last obtained value) with the ground truth for coronal holes. And lastly, in the seventh row, we compared our ground truth for both the coronal holes and active regions with the Multiple Otsu method, where we combined the first and last value that we obtained. When applying the Minimum method and the last value of the Multiple Otsu method, we took out the background, because these methods take the background as a useful region, and this affects the metrics quite significantly. Both of the metrics will be low, because the whole background will be seen as FN.
When looking at the tables, we see that Multiple Otsu gives us the best results, both when only looking at the active regions, and when looking at the coronal holes. Combining those two gives us then the best results. Therefore, for most of this paper, we will work with the first and last value of the Muliple Otsu method.

## Feature Selection
As discussed in the previous chapter, we made a feature vector with the features described. We standardised this vector using the z-score. With this standardised feature vector, we will calculate the correlation between the features. For this, we use the correlation coefficient matrix R. 
<img width="513" alt="Scherm­afbeelding 2024-12-10 om 22 00 45" src="https://github.com/user-attachments/assets/3cece8e3-aa8d-452c-b029-e0108de02f97">  
We decided to remove the sum average, the ASM and the difference entropy from our feature vector. This gives us a new correlation matrix, shown in 3.13.  
<img width="679" alt="Scherm­afbeelding 2024-12-10 om 22 01 44" src="https://github.com/user-attachments/assets/a9cbfb1c-23ca-40e4-b88c-521cffb09b03">

## Principal Component Analysis
Now we have all our data in our feature vector and we have to start interpreting this. We will start with PCA to reduce our dimensions. We would like to apply PCA to our feature vector. To do this, we used the PCA function from Scikit-learn [22]. We gave two components as input and then applied the fit transform method. This method first calculates the PCs and then projects it to two dimensions. You can see these two components plotted in 3.14.  
<img width="489" alt="Scherm­afbeelding 2024-12-10 om 22 02 54" src="https://github.com/user-attachments/assets/6c54bc31-d4aa-493a-8a67-e0b4c62a11b8">

## t-SNE (t-Distributed Stochastic Neighbour Embedding)
To apply t-SNE to our data, we used the t-SNE function from Scikit Learn [22]. We asked for two components and then transformed our data to two dimensions. A plot of these two components can be seen in 3.16. In this plot we can already see three clusters appearing nicely.
<img width="534" alt="Scherm­afbeelding 2024-12-10 om 22 04 23" src="https://github.com/user-attachments/assets/6b04e941-dd94-435d-8a9d-383e6c79cea4">

## Clustering
Following the dimension reduction using PCA or t-SNE, we will now apply clustering on this data. This allows us to separate our data into groups, with the aim to differentiate between the physical meaning of the segmented regions. For example, we would hope to seperate the coronal holes and active regions and ideally also the rim.  
To apply K-means, we utilise the K-Means function from scikit-learn [22]. Looking at the Kneedle method, we got that the Kneedle point lies at five (figure 3.18). So we first applied K-means to the PCA data asking for five clusters, after which we obtained the result from figure 3.19. When observing the visualization of these five clusters (figure 3.24), we can remark that it looks rather chaotic. There is no real division between coronal holes and active regions any more, and it is not immediately obvious what was clustered on. There seems to be a separation based on area and perimeter, but it is not completely clear. Now we can also look at K-means with two, three, four and six clusters, as in figures 3.20. When looking at the visualisations of them figures 3.21, 3.22, 3.23 and 3.25, we observe the same thing, namely that no distinct properties seem to be shared by the elements from the clusters, except perhaps the size of the area. This leads us to suspect that PCA might not be suitable for our intended purposes.  
<img width="482" alt="Scherm­afbeelding 2024-12-10 om 22 05 22" src="https://github.com/user-attachments/assets/752ebc45-86c9-4d09-acb9-2e81da7ddbaa">  
After looking at K-means applied to our PCA data, we also applied it on the t-SNE data. For the Kneedle method, we use the kneed module [1]. We applied K-means and the Kneedle method to our t-SNE data, and we saw that the Kneedle point is three, which could already be predicted when looking at our plot of the t-SNE data without any clustering applied to it (figure 3.16). When plotting our data and asking for three clusters, we get the result shown in figure 3.27
Now the question is what these clusters represent. We can visualise this in two ways, we can use histograms, or we can use the pictures on which we applied thresholding and link the points in each of the clusters to segmented regions of these pictures. We will have a look at the visualisation of the clusters applied to three samples, namely 2012-04-07 at 05:06:07.84, 2015-04-19 at 14:42:06.84 and 2019-12-08 at 22:30:04.85. Based on the visualisations in figures 3.28 and 3.29, we can conclude that cluster ‘0’ represents the bright spots, cluster ‘1’ represents the coronal holes and lastly, cluster ‘2’ represents the active regions.
We can do the same for less and more clusters. When looking at K-means with two clusters (figure 3.31), we lose the cluster that represents the bright points, this cluster is merged with the one for active regions. When looking at the K-means with more than three clusters, we see that some of the bigger clusters are being split up, as can be seen in figure 3.30. When looking at the visualization of four clusters 3.32, we see in 3.32a that one of the coronal holes is misclassified as an active region. In 3.30b we can see why this is. Indeed, there are a few data points in cluster ’3’ that, if we were to manually determine our clusters ourselves, would be more likely to be in cluster ’1’. This problem can also bee seen in 3.27, what shows that our clustering method is fallible. When looking further at K-means with four, five and six clusters, we see that we do not really gain any interesting extra clusters, and the clusters that we get are quite messy and do not give clear classifications (see figures 3.32, 3.33, 3.34).  
<img width="482" alt="Scherm­afbeelding 2024-12-10 om 22 09 29" src="https://github.com/user-attachments/assets/d48f7c09-971f-441e-a7f4-19679a5ea7ae">  
<img width="484" alt="Scherm­afbeelding 2024-12-10 om 22 09 44" src="https://github.com/user-attachments/assets/ae1cf2f1-a441-4533-a0a3-8a97b3bb921f">  
<img width="467" alt="Scherm­afbeelding 2024-12-10 om 22 10 17" src="https://github.com/user-attachments/assets/30209aee-6fa8-47b9-b0b7-a3739d8589d6">
## Conclusion
First of all we needed to learn to work with the data. Then we took our first steps in image processing and learned about image histograms. After that we tested out a plethora of thresholding algorithms, including Yen’s method, Li’s method and adaptive thresholding. Seeing the results of all this different thresholds definitely taught us what did not work. We had a number of difficulties, a lot of them concerning the corona and the background. To find the best thresholding method, we introduced metrics. Eventually concluding that Otsu’s method worked best. After segmentation we started extracting features, starting with researching the GLCM and briefly dabbling with the Gabor filter, though the latter did not look promising. Next we took our first steps in machine learning, initially exploring different standardisation methods. After playing around with all the different parameters of PCA and t-SNE, we looked at the correlation of all our features. Later on we found clusters using K-means and needed to delve in to what they represented.
We found that Otsu’s method worked best and reduction via t-SNE resulted in better clusters using K-means. This project was a journey with ups and downs, we learned a lot, though we made mistakes as well.
### Future Work
For all three of us this was our first time to coming in contact with solar physics, image processing and machine learning. We learned a lot by doing this project and were later made aware of mistakes made.
The first thing we would do is correct these mistakes, when thresholding we were looking at ways to threshold both active regions and coronal holes at the same time. We had found techniques that allowed us to make a mask with only active regions or coronal holes. Later we made masks including both, thus removing the segmentation we already had. It would have been a much wiser decision to work with two separate masks, calculate our features and do machine learning on both independently. We tried fixing this mistake, but this definitely needs more attention.
After fixing these mistakes our framework would need more testing and then expanding. Working with both PCA and t-SNE we discovered that t-SNE worked much better. We got clusters seemingly without any meaning for PCA and we assume this is because the feature ‘area’ influnces the results greatly. While continuing our work, we might find a way to improve this. For example, we could look at other, possibly more complicated features, and their effects on the clusters. During our work, we also noticed that the rim of the sun got classified as an active region, we preferred this to be in a separate cluster, so we added additional features, but this did not give us different results cluster-wise. With additional, or other features we may achieve this. In addition, we could also return to our segmentation technique, in the visualizations of the clusters, it can be noticed that certain large coronal holes and active regions get divided into smaller pieces, this obviously has an effect on the calculation on the features, and thus on the clustering. It might be useful to, instead of asking for 5 classes, try asking for four classes, and therefore three thresholds and see what the effects of this are on the clustering. Afterwards, we could classify the regions and then track how they change. Then we could learn more about how these active regions and coronal holes behave and change.
### Conclusion

In the end, we have developed a framework that is easy to understand and computational friendly. Using the accessible data set from Galvez et al. [8], we applied automatic thresholding methods to distinguish interesting regions. We extracted various features, after which we reduced their dimension using PCA and t-SNE. Using these components, we used K-means clustering to group similar elements. In this way, we can differentiate between detected coronal holes, bright spots and active regions. Using our framework, we can give new data as input and immediately compute interesting regions and their meaning. Since our program does not rely on deep learning techniques, it is computationally easier than most alternatives.

### Acknowledges
We thank the SDO teams for providing the AIA data we used in our project. The dataset introduced in Galvez et al. [8] made our work possible, thus we want to thank all people involved in constructing this dataset. We are grateful for the guidance and assistance given by dr Panagiotis Gonidakis and Francesco Carella and their comments on this report to improve it.
## References
1] Arvai, K. kneed, 2024. Accessed: 2024-11-18.  
[2] Berman, J. J. Chapter 4 - understanding your data. In Data Simplification, J. J.  
Berman, Ed. Morgan Kaufmann, Boston, 2016, pp. 135–187.  
[3] Berry, M. W., Mohamed, A., and Yap, B. W. Supervised and unsupervised learning for data science. Springer, 2019.  
[4] Burger, W., and Burge, M. Digital image processing : an algorithmic introduction, third edition. ed. Texts in computer science. Springer Nature Switzerland AG, Cham, Switzerland, 2022.  
[5] Delouille, V., Hofmeister, S. J., Reiss, M. A., Mampaey, B., Temmer, M., and Veronig, A. Chapter 15 - coronal holes detection using supervised classification. In Machine Learning Techniques for Space Weather, E. Camporeale, S. Wing, and J. R. Johnson, Eds. Elsevier, 2018, pp. 365–395.  
[6] Eda Kavlakoglu, V. W. What is k-means clustering, 2024. Accessed: 2024-11-18.  
[7] Erdem, K. t-sne clearly explained, 2020. Retrieved from https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a.  
[8] Galvez, R., Fouhey, D. F., Jin, M., Szenicer, A., Mun ̃oz-Jaramillo, A., Cheung, M. C. M., Wright, P. J., Bobra, M. G., Liu, Y., Mason, J., and Thomas, R. A Machine-learning Data Set Prepared from the NASA Solar Dynamics Observatory Mission. Astrophysical Journal Supplement Series 242, 1 (May 2019), 7.  
[9] Giakoumoglou, N. G. pyfeats: Image feature extraction tool. Tech. rep., 7 2021. Accessed: 2024-11-16.  
[10] Glasbey, C. An analysis of histogram-based thresholding algorithms. CVGIP: Graphical Models and Image Processing 55, 6 (1993), 532–537.  
[11] Haigh, J. D., Lockwood, M. M., Giampapa, M. S. M. S., and fur Astrophysik und Astronomie., S. G. The sun, solar analogs and the climate, 1st ed. 2005. ed. Saas-Fee advanced course lecture notes ; 2004. Springer-Verlag, Berlin, 2005.  
[12] James Mason, P. W. Sdo machine learning dataset, 2024. Accessed: 2024-11-15.  
[13] Jolliffe, I. T., and Cadima, J. Principal component analysis: A review and recent developments. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 374, 2065 (Apr 2016), 20150202.  
[14] Lemen, J. R., Title, A. M., Akin, D. J., Boerner, P. F., Chou, C., Drake, J. F., Duncan, D. W., Edwards, C. G., Friedlaender, F. M., Heyman, G. F., Hurlburt, N. E., Katz, N. L., Kushner, G. D., Levay, M., Lindgren, R. W., Mathur, D. P., McFeaters, E. L., Mitchell, S., Rehse, R. A., Schrijver, C. J., Springer, L. A., Stern, R. A., Tarbell, T. D., Wuelser, J.-P., Wolfson, C. J., Yanari, C., Bookbinder, J. A., Cheimets, P. N., Caldwell, D., Deluca, E. E., Gates, R., Golub, L., Park, S., Podgor, W. A., Bush, R. I., Scherrer, P. H., Gummin, M. A., Smith, P., Auker, G., Jerram, P., Pool, P., Soufli, R., Windt, D. L., Beardsley, S., Clapp, M., Lang, J., and Waltham, N. The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO). Solar Physics.  
[15] M Pelakhata, K Muraw, S. P. Generation of solar chromosphere heating and coronal outflows by two-fluid waves. Astronomy & Astrophysics (2023).  
[16] Mackovjak, v., Harman, M., Maslej-Kreˇsnˇa ́kova ́, V., and Butka, P. SCSS-Net: solar corona structures segmentation by deep learning. Monthly Notices of the Royal Astronomical Society 508, 3 (10 2021), 3111–3124.  
[17] Narayanan, A. S. An introduction to waves and oscillations in the sun, 1st ed. 2013. ed. Astronomy and astrophysics library. Springer, New York, 2013.  
[18] NASA. Anatomy of the sun, 2024. Public Domain.  
[19] NASA. Sdo — mission: Spacecraft, 2024. Accessed: 2024-11-16.  
[20] NASA/SDO. Sdo: The sun today, 2024. Public Domain.  
[21] Otsu, N. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics 9, 1 (Jan 1979), 62–66.  
[22] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., David, R., Weiss, A., and Dubourg, V. Scikit-learn: Machine learning in python, 2011. Accessed: 2024-11-23.  
[23] Pesnell, W. D., Thompson, B. J., and Chamberlin, P. C. The Solar Dynamics Observatory (SDO). Solar Physics 275, 1-2 (Jan. 2012), 3–15.  
[24] Piech, C. K means, 2013. Accessed: 2024-11-18.  
[25] Powers, D. M. W. Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation. arXiv e-prints (Oct. 2020), arXiv:2010.16061.  
[26] Reiss, M., Temmer, M., Rotter, T., Hofmeister, S. J., and Veronig, A. M. Identification of coronal holes and filament channels in SDO/AIA 193 ̊A images via geometrical classification methods. Central European Astrophysical Bulletin 38 (Jan. 2014), 95–104.  
[27] Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. Finding a ”kneedle” in a haystack: Detecting knee points in system behavior. In 2011 31st International Conference on Distributed Computing Systems Workshops (June 2011), pp. 166–171.  
[28] Schou, J., Scherrer, P. H., Bush, R. I., Wachter, R., Couvidat, S., Rabello-Soares, M. C., Bogart, R. S., Hoeksema, J. T., Liu, Y., Duvall, T. L., Akin, D. J., Allard, B. A., Miles, J. W., Rairden, R., Shine, R. A., Tarbell, T. D., Title, A. M., Wolfson, C. J., Elmore, D. F., Norton, A. A., and Tomczyk, S. Design and Ground Calibration of the Helioseismic and Magnetic Imager (HMI) Instrument on the Solar Dynamics Observatory (SDO). Solar Phyics 275, 1-2 (Jan. 2012), 229–259.  
[29] Van Der Maaten, L., and Hinton, G. Visualizing data using t-sne. Journal of machine learning research 9 (2008), 2579–2625.  
[30] Wentzel, D. G. The restless sun. [Smithsonian library of the solar system]. Smithsonian Institution Press, Washington, 1989.  
[31] Woods, T. N., Eparvier, F. G., Hock, R., Jones, A. R., Woodraska, D., Judge, D., Didkovsky, L., Lean, J., Mariska, J., Warren, H., McMullin, D., Chamberlin, P., Berthiaume, G., Bailey, S., Fuller-Rowell, T., Sojka, J., Tobiska, W. K., and Viereck, R. Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of Science Objectives, Instrument Design, Data Products, and Model Developments. Solar Physics 275, 1-2 (Jan. 2012), 115–143.  
[32] Zarr-Developers. Zarr-python, 2024. Accessed: 2024-11-16.  
[33] Zhang, J., Temmer, M., Gopalswamy, N., Malandraki, O., Nitta, N. V., Patsourakos, S., Shen, F., Vrˇsnak, B., Wang, Y., Webb, D., Desai, M. I., Dissauer, K., Dresing, N., Dumbovic ́, M., Feng, X., Heinemann, S. G., Laurenza, M., Lugaz, N., and Zhuang, B. Earth-affecting solar transients: a review of progresses in solar cycle 24. Progress in earth and planetary science 8, 1 (2021), 56–56.  
[34] Zhang, T., Hao, Q., and Chen, P. F. Statistical analyses of solar prominences and active region features in 304  ̊a filtergrams detected via deep learning, 2024.  
