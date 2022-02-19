# HarmonizationPaper
The goal of this work is to attempt to establish the minimum sample size required for achieving inter-site harmonization using features derived from structural brain MRI. Leveraging the concept of learning curves, we provide a framework of Mahalanobis distance as a way to understand site-effects using real and simulated data.

## Experiment 1
The goal of this experiment is to ask "are there differences in the distribution of features (univariate) from different scanners using a Kolmogorov–Smirnov test?" and whether these differences remain after harmonization. This can be implemented using `kstest` in MATLAB.

## Experiment 2
The goal of this experiment is to ask "can we predict (above chance level) which scanner/site the data comes from?" 
In the first part of this experiment, in a cross-validation framework, we train and test a linear SVM classifier and ask the question: "if the data is not harmonized, can we learn to predict site better than chance?" 
In the second part of this experiment, in a cross-validation framework, we train and test a linear SVM classifier and ask the question: "if the data is harmonized, can we still learn to predict site better than chance?"

## Experiment 3
The goal of this experiment is to draw a series of learning curves; for a given pool of data, we split the data into k folds - SVMTest set and the remaining data; the remaining data is then split into SVMTrain set and the NHLearn set. Within the NHLearn set, we sub-select different sample sizes (as specified in allCurves) and use them to learn harmonization parameters. These harmonization parameters are then applied to SVMTrain and SVMTest sets. After this a linear SVM classifier is trained using the features in SVMTrain and then predictions are made on SVMTest. Permutation testing is performed to test if the classifier performance is above chance level

## Experiment 4
The goal of this experiment is to draw a series of learning curves using simulated data; for a given pool of data, we split the data into 20 folds - SVMTest set and the remaining data; the remaining data is then split into SVMTrain set (70 samples per site) and the NHLearn set. Within the NHLearn set, we sub-select different sample sizes (as specified in 
allCurves) and use them to learn harmonization parameters. These harmonization parameters are then applied to SVMTrain and SVMTest sets. After this a linear SVM classifier is trained using the features in SVMTrain and then predictions are made on SVMTest. Permutation testing is performed to test if the classifier performance is above chance level

## Supporting functions
### calc_MD
This function returns the calculated Mahalanobis distance between the individual site-distributions and the reference distribution.
First, a reference distribution is created by calculating the average of average of features across all sites (grand mean) and the pooled covariance matrix (which is the average covariance matrix across sites). Next, for each site, we calculate the Mahalanobis distance from this reference distribution.

### neuroHarmonize wrapper functions
There are three wrapper Python functions which call [neuroHarmonize](https://github.com/rpomponio/neuroHarmonize/)

## Data format
The main input variable is a table type variable having the following columns:
* names:    (not used)
* Age:      age of the subject
* Sex:      (not used)
* Female:   0/1 coded for female = yes
* TIV:      TIV of the subject
* Site:     cell type having site which will be used for classification
* Stratify: (not used)
* Actual features start from column 8 onwards

## Citation
The manuscript is under preparation:
Parekh P., Bhalerao G.V., John J.P., Venkatasubramanian G., and the ADBS consortium. Sample size requirement for achieving multisite harmonization using structural brain MRI features. 
