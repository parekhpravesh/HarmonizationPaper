from neuroHarmonize import harmonizationLearn, harmonizationApply, loadHarmonizationModel
import pandas as pd
import numpy as np
import sys

# load your data and all numeric covariates
fullpath  = sys.argv[1]
foldStr   = sys.argv[2]
trainData = pd.read_csv(fullpath.strip() + '/' + foldStr.strip() + '_temp_trainData.csv', header=None, index_col=None)
trainData = np.array(trainData)
covars    = pd.read_csv(fullpath.strip() + '/' + foldStr.strip() + '_temp_trainSite.csv')

# run harmonization and store the adjusted data
model, trainData_adj = harmonizationLearn(trainData, covars)

# Apply model
testData     = pd.read_csv(fullpath.strip() + '/' + foldStr.strip() + '_temp_testData.csv', header=None, index_col=None)
testData     = np.array(testData)
testcovars   = pd.read_csv(fullpath.strip() + '/' + foldStr.strip() + '_temp_testSite.csv')
testData_adj = harmonizationApply(testData, testcovars, model)

# Write out as csv file
trainData_adj = pd.DataFrame(trainData_adj)
testData_adj  = pd.DataFrame(testData_adj)
trainData_adj.to_csv(fullpath.strip() + '/' + foldStr.strip() + '_adjustedTrainData.csv', header=False, index=False)
testData_adj.to_csv(fullpath.strip()  + '/' + foldStr.strip() + '_adjustedTestData.csv',  header=False, index=False)
