function exp002_SVMclassification(dataTable,  regressTIV, regressAge, ...
                                  regressSex, numCV,      numRepeats, ...
                                  numPerms,   Seeds,      outDir)
% Function to predict site using a linear SVM classifier
%% Inputs:
% dataTable:    table type variable; should have the following variables 
%               (in this order):
%                   * names:    subject name
%                   * Age:      age of the subject
%                   * Sex:      character type M/F
%                   * Female:   0/1 coded for female = yes
%                   * TIV:      TIV of the subject
%                   * Site:     cell type having site which will be used 
%                               for classification
%                   * Stratify: cell type having a variable to be used for
%                               stratification during cross-validation
%                   * Actual features start from column 8 onwards
% regressTIV:   true or false indicating if TIV should be regressed
% regressAge:   true or false indicating if age should be regressed
% regressSex:   true or false indicating if sex should be regressed
% numCV:        number of cross-validation splits
% numRepeats:   number of times cross-validation should be done
% numPerms:     number of times permutation testing should be done
% Seeds:        [numPerms x 1] vector of seeds to be used during
%               cross-validation and permutation testing
% outDir:       full path to where results should be saved
% 
%% Output:
% A 'Results.mat' file is saved in the outDir containing the results 
% 
%% Notes:
% The goal of this experiment is to ask "can we predict (above chance level) 
% which scanner/site the data comes from?"
% 
% In the first part of this experiment, in a cross-validation framework, we
% train and test a linear SVM classifier and ask the question: "if the data
% is not harmonized, can we learn to predict site better than chance?"
% 
% In the second part of this experiment, in a cross-validation framework, 
% we train and test a linear SVM classifier and ask the question: "if the 
% data is harmonized, can we still learn to predict site better than 
% chance?"
% 
% To reduce computational cost, we do the following:
% For every repeat
%   For every fold
%       Train and test site prediction on non-harmonized data
%       Permutation test
%       Harmonize data
%       Train and test site prediction on harmonized data
%       Permutation test
% 
% For remaining repeats (assuming permutation repeats > CV repetas)
%   For every fold
%       Permutation test
%       Harmonize data
%       Permutation test
% 
% Given this framework, we only harmonize the data once thereby (hopefully)
% reducing the computational time dramatically
% 
% For permutation testing of harmonized data, we harmonize the data 
% correctly but permute the labels just before SVM to test if the site 
% prediction accuracies are above chance (rather than permuting first and 
% then harmonizing the data)
% 
%% Defaults:
% regressTIV:   true
% regressAge:   true
% regressSex:   true
% numCV:        10
% numRepeats:   50
% numPerms:     100
% Seeds:        rng(0, 'twister'); randi(9999999, numPerms, 1)
% outDir:       pwd/exp002
% 
%% Authors:
% Bhalerao, Gaurav
% Parekh, Pravesh
% October 18, 2021
% ADBS

%% Check inputs and assign defaults                                    
% Check dataTable
if ~exist('dataTable', 'var') || isempty(dataTable)
    error('Please provide dataTable to work with');
else
    if ~istable(dataTable)
        error('Expect dataTable to be of table type');
    end
end

% Check regressTIV
if ~exist('regressTIV', 'var') || isempty(regressTIV)
    regressTIV = true;
else
    if ~islogical(regressTIV)
        error('Expect regressTIV to be either true or false');
    end
end

% Check regressAge
if ~exist('regressAge', 'var') || isempty(regressAge)
    regressAge = true;
else
    if ~islogical(regressAge)
        error('Expect regressAge to be either true or false');
    end
end

% Check regressSex
if ~exist('regressSex', 'var') || isempty(regressSex)
    regressSex = true;
else
    if ~islogical(regressSex)
        error('Expect regressSex to be either true or false');
    end
end

% Check numCV
if ~exist('numCV', 'var') || isempty(numCV)
    numCV = 10;
end

% Check numRepeats
if ~exist('numRepeats', 'var') || isempty(numRepeats)
    numRepeats = 50;
end

% Check numPerms
if ~exist('numPerms', 'var') || isempty(numPerms)
    numPerms = 100;
end

% Check Seeds
if ~exist('Seeds', 'var') || isempty(Seeds)
    rng(0, 'twister');
    Seeds = randi(9999999, numPerms, 1);
else
    if length(Seeds) ~= numPerms
        error(['Expected ', num2str(numPerms), ' many seeds']);
    end
end

% Check outDir
if ~exist('outDir', 'var') || isempty(outDir)
    outDir = fullfile(pwd, 'exp002');
end
% Make output directory, if it does not exist
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Check if it is a multi class problem
if length(unique(dataTable.Site)) > 2
    multiClass = true;
else
    multiClass = false;
end

%% Perform unsupervised feature elimination
% a) remove constant features
% b) remove features with NaN
% c) remove features with less than 10% unique values
toWork  = dataTable{:, 8:end};

% Find constant locations
locs_constant = find(var(toWork) == 0);

% Find locations which have NaN values
locs_NaN = find(sum(isnan(toWork)) ~= 0);

% Find locations which have less than 10% unique values in data 
tmp             = round(toWork, 4);
cutoff          = round(10*size(toWork,1)/100, 0);
count           = 1;
locs_novariance = [];
for feat = 1:size(toWork,2)
    tmp2 = unique(tmp(:,feat));
    if length(tmp2) < cutoff
        locs_novariance(count) = feat; %#ok<AGROW>
        count = count + 1;
    end
end

% All locations to delete: toWork is a matrix
locDelete = unique([locs_constant'; locs_NaN'; locs_novariance']);

% Actual locations to delete in the original table
locDelete = locDelete + 7;

% Record variable names that are being removed
deletedVariables = dataTable.Properties.VariableNames(locDelete); %#ok<NASGU>

% Delete these variables
dataTable(:, locDelete) = [];

%% Initialize
tmpCV                           = cvpartition(dataTable.Site, 'KFold', numCV, 'stratify', true);
allTrainActSite                 = cell(max(tmpCV.TrainSize), numCV, numRepeats);
allTestActSite                  = cell(max(tmpCV.TestSize),  numCV, numRepeats);
allTrainActSite_Perm            = cell(max(tmpCV.TrainSize), numCV, numPerms);
allTestActSite_Perm             = cell(max(tmpCV.TestSize),  numCV, numPerms);

allTrainPredictions_raw         = cell(max(tmpCV.TrainSize), numCV, numRepeats);
allTestPredictions_raw          = cell(max(tmpCV.TestSize),  numCV, numRepeats);
allTrainPredictions_raw_Perm    = cell(max(tmpCV.TrainSize), numCV, numPerms);
allTestPredictions_raw_Perm     = cell(max(tmpCV.TestSize),  numCV, numPerms);

allTrainPredictions_harm        = cell(max(tmpCV.TrainSize), numCV, numRepeats);
allTestPredictions_harm         = cell(max(tmpCV.TestSize),  numCV, numRepeats);
allTrainPredictions_harm_Perm   = cell(max(tmpCV.TrainSize), numCV, numPerms);
allTestPredictions_harm_Perm    = cell(max(tmpCV.TestSize),  numCV, numPerms);

performanceTrain_raw            = cell(numRepeats, 4);
performanceTest_raw             = cell(numRepeats, 4);
performanceTrain_raw_Perm       = cell(numPerms, 4);
performanceTest_raw_Perm        = cell(numPerms, 4);

performanceTrain_harm           = cell(numRepeats, 4);
performanceTest_harm            = cell(numRepeats, 4);
performanceTrain_harm_Perm      = cell(numPerms, 4);
performanceTest_harm_Perm       = cell(numPerms, 4);

varNames = {'RepeatNumber', 'MeanAccuracy', 'StdAccuracy', 'Accuracy_MeanSD'};

%% ML pipeline: 1 to numRepeats
count       = 1;
skipCV      = false;
for repeats = 1:numRepeats
    
    % Start timer
    t_init = tic;
    
    % Run CV pipeline
    [allTrainActSite(:, :, count),                allTestActSite(:, :, count),                 ...
     allTrainActSite_Perm(:, :, count),           allTestActSite_Perm(:, :, count),            ...
     allTrainPredictions_raw(:, :, count),        allTestPredictions_raw(:, :, count),         ...
     allTrainPredictions_raw_Perm(:, :, count),   allTestPredictions_raw_Perm(:, :, count),    ...
     allTrainPredictions_harm(:, :, count),       allTestPredictions_harm(:, :, count),        ...
     allTrainPredictions_harm_Perm(:, :, count),  allTestPredictions_harm_Perm(:, :, count),   ...
     performanceTrain_raw(count, :),              performanceTest_raw(count, :),               ...
     performanceTrain_raw_Perm(count, :),         performanceTest_raw_Perm(count, :),          ...
     performanceTrain_harm(count, :),             performanceTest_harm(count, :),              ...
     performanceTrain_harm_Perm(count, :),        performanceTest_harm_Perm(count, :)] =       ...
     withinCV(Seeds(count), numCV, dataTable, regressTIV, regressAge,                          ...
              regressSex, multiClass, count, skipCV, outDir);
          
     % Stop timer
     t_end = toc(t_init);
         
    % Update user
    disp(['Finished Repeat: ', num2str(count, '%02d'), ' [', num2str(t_end, '%.2f'), ' seconds]']);
          
     % Update counter
     count = count + 1;
end

%% ML pipeline: numRepeats+1 to numPerms
skipCV      = true;
for repeats = count:numPerms
    
    % Start timer
    t_init = tic;    
    
    % Run CV pipeline
    [~,                                           ~,                                           ...
     allTrainActSite_Perm(:, :, count),           allTestActSite_Perm(:, :, count),            ...
     ~,                                           ~,                                           ...
     allTrainPredictions_raw_Perm(:, :, count),   allTestPredictions_raw_Perm(:, :, count),    ...
     ~,                                           ~,                                           ...
     allTrainPredictions_harm_Perm(:, :, count),  allTestPredictions_harm_Perm(:, :, count),   ...
     ~,                                           ~,                                           ...
     performanceTrain_raw_Perm(count, :),         performanceTest_raw_Perm(count, :),          ...
     ~,                                           ~,                                           ...
     performanceTrain_harm_Perm(count, :),        performanceTest_harm_Perm(count, :)] =       ...
     withinCV(Seeds(count), numCV, dataTable, regressTIV, regressAge,                          ...
              regressSex, multiClass, count, skipCV, outDir);
          
     % Stop timer
     t_end = toc(t_init);
         
    % Update user
    disp(['Finished Repeat: ', num2str(count, '%02d'), ' [', num2str(t_end, '%.2f'), ' seconds]']);          
          
     % Update counter
     count = count + 1;          
end

%% Convert to tables
performanceTrain_raw        = cell2table(performanceTrain_raw,        'VariableNames', varNames); %#ok<NASGU>
performanceTest_raw         = cell2table(performanceTest_raw,         'VariableNames', varNames);
performanceTrain_raw_Perm   = cell2table(performanceTrain_raw_Perm,   'VariableNames', varNames); %#ok<NASGU>
performanceTest_raw_Perm    = cell2table(performanceTest_raw_Perm,    'VariableNames', varNames);

performanceTrain_harm       = cell2table(performanceTrain_harm,       'VariableNames', varNames); %#ok<NASGU>
performanceTest_harm        = cell2table(performanceTest_harm,        'VariableNames', varNames);
performanceTrain_harm_Perm  = cell2table(performanceTrain_harm_Perm,  'VariableNames', varNames); %#ok<NASGU>
performanceTest_harm_Perm   = cell2table(performanceTest_harm_Perm,   'VariableNames', varNames);

%% Compute p values
% For each original accuracy, count the number of times the permutation
% accuracy became equal to or exceeded this value
% Alternatively, error is lesser in permutation set
pValue_raw_test  = zeros(numRepeats,1);
pValue_harm_test = zeros(numRepeats,1);

for repeat = 1:numRepeats
    pValue_raw_test(repeat, 1)  = (sum(performanceTest_raw_Perm.MeanAccuracy  >= performanceTest_raw.MeanAccuracy(repeat))  + 1)/(numPerms + 1);
    pValue_harm_test(repeat, 1) = (sum(performanceTest_harm_Perm.MeanAccuracy >= performanceTest_harm.MeanAccuracy(repeat)) + 1)/(numPerms + 1);
end

% Average p values
overallpValue_raw_test  = mean(pValue_raw_test);  %#ok<NASGU>
overallpValue_harm_test = mean(pValue_harm_test); %#ok<NASGU>

% Append p Value to each repeat
performanceTest_raw.pValue  = pValue_raw_test;
performanceTest_harm.pValue = pValue_harm_test;

%% Save everything
save(fullfile(outDir, 'Results.mat'));
% -------------------------------------------------------------------------

function [stdData, stdCoeff] = standardizeData(data, stdCoeff)
% Function to standardize a given feature matrix or apply already learned
% standardization coefficients to a feature matrix
% stdData  = (data - mean(data))/std(data)
% stdCoeff = [mean(data), std(data)]

% Determine if standardization coefficients are o be learnt or not
if ~exist('stdCoeff', 'var') || isempty(stdCoeff)
    toEstimate = true;
else
    toEstimate = false;
end

% Learn coefficients
if toEstimate
    stdCoeff(1,:) = mean(data);
    stdCoeff(2,:) = std(data);
end
    
% Apply scaling
% stdData = (data - stdCoeff(1))./ stdCoeff(2);
stdData = bsxfun(@rdivide, bsxfun(@minus, data, stdCoeff(1,:)), stdCoeff(2,:));
% -------------------------------------------------------------------------

function [regressedData, coeff] = regressVariables(matrix, covariates, coeff)
% Function to regress covariates from each column of a matrix
% Automatically adds a constant term as the last column of covariates

% Add constant term to covariates
covariates(:, end+1) = ones(size(matrix,1),1);

% Number of features, number of observations, and number of covariates
numFeatures     = size(matrix,2);
numObservations = size(matrix,1);
numCovariates   = size(covariates,2);

% Check if regression coefficient should be learnt
if ~exist('coeff', 'var') || isempty(coeff)
    to_estimate = true;
else
    to_estimate = false;
end

% Estimate coefficients, if necessary
if to_estimate
    coeff = zeros(numFeatures, numCovariates);
    for feat = 1:size(matrix,2)
        coeff(feat, 1:numCovariates) = covariates\matrix(:,feat);
    end
end

% Do regression and calculate residuals
regressedData = zeros(numObservations, numFeatures);
for feat      = 1:numFeatures
    yhat      = sum(bsxfun(@times, coeff(feat,1:numCovariates), covariates),2);
    regressedData(:,feat) = matrix(:,feat) - yhat;
end
% -------------------------------------------------------------------------

function [hTrainData, hTestData] = doHarmonization(trainData, testData, ...
                                                   trainSite, testSite, ...
                                                   trainCov,  testCov,  ...
                                                   covNames,  tmpDir, foldNum)
% Function to call neuroHarmonize and return harmonized data

% Fold number
foldStr = ['fold_', num2str(foldNum, '%02d')];

% Write out csv files
dlmwrite(fullfile(tmpDir, [foldStr, '_temp_trainData.csv']), trainData);
dlmwrite(fullfile(tmpDir, [foldStr, '_temp_testData.csv']),  testData);

% Put covariates together
trainSite = [trainSite, num2cell(trainCov)];
testSite  = [testSite,  num2cell(testCov)];

% Create covariates files for writing
fid_train = fopen(fullfile(tmpDir, [foldStr, '_temp_trainSite.csv']), 'w');
fid_test  = fopen(fullfile(tmpDir, [foldStr, '_temp_testSite.csv']),  'w');

% Print header
% Walter's example: https://www.mathworks.com/matlabcentral/answers/364295
tmpHeader = ['SITE', covNames];   
fprintf(fid_train, '%s,',  tmpHeader{1:end-1});
fprintf(fid_train, '%s\n', tmpHeader{end});
fprintf(fid_test,  '%s,',  tmpHeader{1:end-1});
fprintf(fid_test,  '%s\n', tmpHeader{end});

% Write out data: integer or float doesn't seem to matter
for lines = 1:size(trainSite,1)
    fprintf(fid_train, '%s,%f,%f,%f\n', trainSite{lines,:});
end
for lines = 1:size(testSite,1)
    fprintf(fid_test, '%s,%f,%f,%f\n', testSite{lines,:});
end

% Close files
fclose(fid_train);
fclose(fid_test);

% Write out covariates files: depracated (slow!)
% trainSite = cell2table(trainSite, 'VariableNames', ['SITE', covNames]);
% testSite  = cell2table(testSite,  'VariableNames', ['SITE', covNames]);
% writetable(trainSite, fullfile(tmpDir, [foldStr, '_temp_trainSite.csv']));
% writetable(testSite,  fullfile(tmpDir, [foldStr, '_temp_testSite.csv']));

% Might need to add full path to doHarmonization.py script
command = ['python doHarmonization.py ', tmpDir, ' ', foldStr];
system(command);

% Read adjusted data back in
hTrainData = dlmread(fullfile(tmpDir, [foldStr, '_adjustedTrainData.csv']));
hTestData  = dlmread(fullfile(tmpDir, [foldStr, '_adjustedTestData.csv']));

% Delete files
delete(fullfile(tmpDir, [foldStr, '_temp_trainData.csv']));
delete(fullfile(tmpDir, [foldStr, '_temp_testData.csv']));
delete(fullfile(tmpDir, [foldStr, '_temp_trainSite.csv']));
delete(fullfile(tmpDir, [foldStr, '_temp_testSite.csv']));
delete(fullfile(tmpDir, [foldStr, '_adjustedTrainData.csv']));
delete(fullfile(tmpDir, [foldStr, '_adjustedTestData.csv']));
% -------------------------------------------------------------------------

function [groundTruth_Train,            groundTruth_Test,              ...
          groundTruth_Train_Perm,       groundTruth_Test_Perm,         ...
          prediction_raw_Train,         prediction_raw_Test,           ...
          prediction_raw_Train_Perm,    prediction_raw_Test_Perm,      ...
          prediction_harm_Train,        prediction_harm_Test,          ...
          prediction_harm_Train_Perm,   prediction_harm_Test_Perm,     ...
          performanceTrain_raw,         performanceTest_raw,           ...
          performanceTrain_raw_Perm,    performanceTest_raw_Perm,      ...
          performanceTrain_harm,        performanceTest_harm,          ...          
          performanceTrain_harm_Perm,   performanceTest_harm_Perm] =   ...
          withinCV(seed, numCV, dataTable, regressTIV, regressAge,     ...
                   regressSex, multiClass, repNumber, skipCV, outDir)

% This function performs the nitty gritty details of cross validation and
% returns training and test mean, standard deviation, and text version of 
% mean and standard deviation of accuracies; returned groundTruth and 
% prediction are fold-wise (cell type)

% Set seed 
rng(seed, 'twister');
    
% Make a working version of the dataTable and work with this
workTable = dataTable;

% Generate a cvpartition object for this repeat
cv = cvpartition(workTable.Site, 'KFold', numCV, 'stratify', true);

% Initialize
groundTruth_Train            = cell(max(cv.TrainSize), numCV);
groundTruth_Test             = cell(max(cv.TestSize),  numCV);
groundTruth_Train_Perm       = cell(max(cv.TrainSize), numCV);
groundTruth_Test_Perm        = cell(max(cv.TestSize),  numCV);

prediction_raw_Train         = cell(max(cv.TrainSize), numCV);
prediction_raw_Test          = cell(max(cv.TestSize),  numCV);
prediction_raw_Train_Perm    = cell(max(cv.TrainSize), numCV);
prediction_raw_Test_Perm     = cell(max(cv.TestSize),  numCV);

prediction_harm_Train        = cell(max(cv.TrainSize), numCV);
prediction_harm_Test         = cell(max(cv.TestSize),  numCV);
prediction_harm_Train_Perm   = cell(max(cv.TrainSize), numCV);
prediction_harm_Test_Perm    = cell(max(cv.TestSize),  numCV);

foldwiseTrainAcc_raw         = zeros(cv.NumTestSets,   1);
foldwiseTestAcc_raw          = zeros(cv.NumTestSets,   1);
foldwiseTrainAcc_raw_Perm    = zeros(cv.NumTestSets,   1);
foldwiseTestAcc_raw_Perm     = zeros(cv.NumTestSets,   1);

foldwiseTrainAcc_harm        = zeros(cv.NumTestSets,   1);
foldwiseTestAcc_harm         = zeros(cv.NumTestSets,   1);
foldwiseTrainAcc_harm_Perm   = zeros(cv.NumTestSets,   1);
foldwiseTestAcc_harm_Perm    = zeros(cv.NumTestSets,   1);

% Go over folds
for fold = 1:numCV
        
    % Training and test data split
    dataTrain = workTable(cv.training(fold),:);
    dataTest  = workTable(cv.test(fold),    :);
        
    % Split into features and class labels
    featuresTrain_raw = dataTrain{:, 8:end};
    labelsTrain       = dataTrain.Site;
    featuresTest_raw  = dataTest{:,  8:end};
    labelsTest        = dataTest.Site;
    
    % Prepare covariates: at this stage, mandatory
    covariatesTrain  = [dataTrain.TIV, dataTrain.Age, dataTrain.Female];
    covariatesTest   = [dataTest.TIV,  dataTest.Age,  dataTest.Female];
    covNames         = {'TIV', 'Age', 'Female'};
    
    % Additionally prepare harmonized data
    [featuresTrain_harm, featuresTest_harm] = doHarmonization(featuresTrain_raw, featuresTest_raw,  ...
                                                              labelsTrain,       labelsTest,        ...
                                                              covariatesTrain,   covariatesTest,    ...
                                                              covNames,          outDir, fold);

    % Prepare labels for permutation testing
    toGen             = length(labelsTrain) + length(labelsTest);
    toReorder         = randperm(toGen, toGen);
    labelsAll         = [labelsTrain; labelsTest];
    labelsAll         = labelsAll(toReorder);
    labelsTrain_Perm  = labelsAll(cv.training(fold));
    labelsTest_Perm   = labelsAll(cv.test(fold));
                 
    % Regress covariates, if needed from both raw and harmonized data
    toRegress        = [regressTIV,    regressAge,    regressSex];
    covariatesTrain  = covariatesTrain(:, toRegress);
    covariatesTest   = covariatesTest(:,  toRegress);
    if ~isempty(covariatesTrain)
        % Raw data
        [featuresTrain_raw, coeff]   = regressVariables(featuresTrain_raw, covariatesTrain);
        featuresTest_raw             = regressVariables(featuresTest_raw,  covariatesTest, coeff);
        
        % Harmonized data
        [featuresTrain_harm, coeff]  = regressVariables(featuresTrain_harm, covariatesTrain);
        featuresTest_harm            = regressVariables(featuresTest_harm,  covariatesTest, coeff);
    end
    
    % Standardize raw and harmonized data
    % Raw data
    [featuresTrain_raw, std_params]  = standardizeData(featuresTrain_raw);
    featuresTest_raw                 = standardizeData(featuresTest_raw, std_params);
    
    % Harmonized data
    [featuresTrain_harm, std_params] = standardizeData(featuresTrain_harm);
    featuresTest_harm                = standardizeData(featuresTest_harm, std_params);
    
    % Train linear SVM on raw and harmonized data
    if multiClass
        if ~skipCV
            mdl_raw     = fitcecoc(featuresTrain_raw,  labelsTrain, 'Coding', 'onevsone', 'Learners', 'svm');
            mdl_harm    = fitcecoc(featuresTrain_harm, labelsTrain, 'Coding', 'onevsone', 'Learners', 'svm');            
        end
        mdl_raw_Perm    = fitcecoc(featuresTrain_raw,  labelsTrain_Perm, 'Coding', 'onevsone', 'Learners', 'svm');
        mdl_harm_Perm   = fitcecoc(featuresTrain_harm, labelsTrain_Perm, 'Coding', 'onevsone', 'Learners', 'svm');
    else
        if ~skipCV
            mdl_raw     = fitcsvm(featuresTrain_raw,   labelsTrain,  'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
            mdl_harm    = fitcsvm(featuresTrain_harm,  labelsTrain,  'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
        end
        mdl_raw_Perm    = fitcsvm(featuresTrain_raw,   labelsTrain_Perm,  'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
        mdl_harm_Perm   = fitcsvm(featuresTrain_harm,  labelsTrain_Perm,  'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
    end
    
    % Record ground truth
    groundTruth_Train(1:length(labelsTrain), fold)           = labelsTrain;
    groundTruth_Test(1:length(labelsTest),   fold)           = labelsTest;
    
    % Record permutation type 1 ground truth
    groundTruth_Train_Perm(1:length(labelsTrain), fold)      = labelsTrain_Perm;
    groundTruth_Test_Perm(1:length(labelsTest),   fold)      = labelsTest_Perm;
        
    if ~skipCV
        % Record predictions - raw, actual data
        prediction_raw_Train(1:length(labelsTrain), fold)    = predict(mdl_raw, featuresTrain_raw);
        prediction_raw_Test(1:length(labelsTest),   fold)    = predict(mdl_raw, featuresTest_raw);
        
        % Record predictions - harmonized, actual data
        prediction_harm_Train(1:length(labelsTrain), fold)   = predict(mdl_harm, featuresTrain_harm);
        prediction_harm_Test(1:length(labelsTest),   fold)   = predict(mdl_harm, featuresTest_harm);
    end
    
    % Record predictions - raw, permutation
    prediction_raw_Train_Perm(1:length(labelsTrain), fold)  = predict(mdl_raw_Perm, featuresTrain_raw);
    prediction_raw_Test_Perm(1:length(labelsTest),   fold)  = predict(mdl_raw_Perm, featuresTest_raw);
       
    % Record predictions - harmonized, permutation
    prediction_harm_Train_Perm(1:length(labelsTrain), fold) = predict(mdl_harm_Perm, featuresTrain_harm);
    prediction_harm_Test_Perm(1:length(labelsTest),   fold) = predict(mdl_harm_Perm, featuresTest_harm);
        
    if ~skipCV
        
        % Identify empty rows (if any) and ignore them
        locEmpty_Train = cellfun(@isempty, groundTruth_Train(:, fold));
        locEmpty_Test  = cellfun(@isempty, groundTruth_Test(:,  fold));

        % Fold-wise accuracies - raw, actual data
        foldwiseTrainAcc_raw(fold, 1)    = sum(strcmpi(groundTruth_Train(~locEmpty_Train, fold), prediction_raw_Train(~locEmpty_Train, fold)))/length(prediction_raw_Train(~locEmpty_Train, fold))*100;
        foldwiseTestAcc_raw(fold,  1)    = sum(strcmpi(groundTruth_Test(~locEmpty_Test,   fold), prediction_raw_Test(~locEmpty_Test,  fold)))/length(prediction_raw_Test(~locEmpty_Test,  fold))*100;
        
        % Fold-wise accuracies - harmonized, actual data
        foldwiseTrainAcc_harm(fold, 1)   = sum(strcmpi(groundTruth_Train(~locEmpty_Train, fold), prediction_harm_Train(~locEmpty_Train, fold)))/length(prediction_harm_Train(~locEmpty_Train, fold))*100;
        foldwiseTestAcc_harm(fold,  1)   = sum(strcmpi(groundTruth_Test(~locEmpty_Test,  fold),  prediction_harm_Test(~locEmpty_Test,  fold)))/length(prediction_harm_Test(~locEmpty_Test,  fold))*100;
    end
    
    % Fold-wise accuracies - raw, permutation
    % Identify empty rows (if any) and ignore them
    locEmpty_Train = cellfun(@isempty, groundTruth_Train_Perm(:, fold));
    locEmpty_Test  = cellfun(@isempty, groundTruth_Test_Perm(:,  fold));
    
    foldwiseTrainAcc_raw_Perm(fold, 1)  = sum(strcmpi(groundTruth_Train_Perm(~locEmpty_Train, fold), prediction_raw_Train_Perm(~locEmpty_Train, fold)))/length(prediction_raw_Train_Perm(~locEmpty_Train, fold))*100;
    foldwiseTestAcc_raw_Perm(fold,  1)  = sum(strcmpi(groundTruth_Test_Perm(~locEmpty_Test,  fold),  prediction_raw_Test_Perm(~locEmpty_Test,   fold)))/length(prediction_raw_Test_Perm(~locEmpty_Test,   fold))*100;
    
    % Fold-wise accuracies - harmonized, permutation
    foldwiseTrainAcc_harm_Perm(fold, 1) = sum(strcmpi(groundTruth_Train_Perm(~locEmpty_Train, fold), prediction_harm_Train_Perm(~locEmpty_Train, fold)))/length(prediction_harm_Train_Perm(~locEmpty_Train, fold))*100;
    foldwiseTestAcc_harm_Perm(fold,  1) = sum(strcmpi(groundTruth_Test_Perm(~locEmpty_Test,   fold), prediction_harm_Test_Perm(~locEmpty_Test,   fold)))/length(prediction_harm_Test_Perm(~locEmpty_Test,   fold))*100;    
end

% Summarize repeat number, mean accuracy, standard deviation of accuracy, 
% and mean and standard deviation
if ~skipCV
    % Model performance: average over folds - raw, actual data
    performanceTrain_raw{1, 1} = repNumber;
    performanceTrain_raw{1, 2} = mean(foldwiseTrainAcc_raw);
    performanceTrain_raw{1, 3} = std(foldwiseTrainAcc_raw);
    performanceTrain_raw{1, 4} = [num2str(performanceTrain_raw{1,2}, '%.2f'), ' ± ', num2str(performanceTrain_raw{1,3}, '%.2f')];

    performanceTest_raw{1,  1} = repNumber;
    performanceTest_raw{1,  2} = mean(foldwiseTestAcc_raw);
    performanceTest_raw{1,  3} = std(foldwiseTestAcc_raw);
    performanceTest_raw{1,  4} = [num2str(performanceTest_raw{1, 2}, '%.2f'), ' ± ', num2str(performanceTest_raw{1, 3}, '%.2f')];

    % Model performance: average over folds - harmonized, actual data
    performanceTrain_harm{1, 1} = repNumber;
    performanceTrain_harm{1, 2} = mean(foldwiseTrainAcc_harm);
    performanceTrain_harm{1, 3} = std(foldwiseTrainAcc_harm);
    performanceTrain_harm{1, 4} = [num2str(performanceTrain_harm{1,2}, '%.2f'), ' ± ', num2str(performanceTrain_harm{1,3}, '%.2f')];

    performanceTest_harm{1,  1} = repNumber;
    performanceTest_harm{1,  2} = mean(foldwiseTestAcc_harm);
    performanceTest_harm{1,  3} = std(foldwiseTestAcc_harm);
    performanceTest_harm{1,  4} = [num2str(performanceTest_harm{1, 2}, '%.2f'), ' ± ', num2str(performanceTest_harm{1, 3}, '%.2f')];
else
    % Set these to NaN
    performanceTrain_raw{1, 1} = NaN;
    performanceTrain_raw{1, 2} = NaN;
    performanceTrain_raw{1, 3} = NaN;
    performanceTrain_raw{1, 4} = NaN;

    performanceTest_raw{1,  1} = NaN;
    performanceTest_raw{1,  2} = NaN;
    performanceTest_raw{1,  3} = NaN;
    performanceTest_raw{1,  4} = NaN;

    performanceTrain_harm{1, 1} = NaN;
    performanceTrain_harm{1, 2} = NaN;
    performanceTrain_harm{1, 3} = NaN;
    performanceTrain_harm{1, 4} = NaN;

    performanceTest_harm{1,  1} = NaN;
    performanceTest_harm{1,  2} = NaN;
    performanceTest_harm{1,  3} = NaN;
    performanceTest_harm{1,  4} = NaN;
end

% Model performance: average over folds - raw, permutation
performanceTrain_raw_Perm{1, 1} = repNumber;
performanceTrain_raw_Perm{1, 2} = mean(foldwiseTrainAcc_raw_Perm);
performanceTrain_raw_Perm{1, 3} = std(foldwiseTrainAcc_raw_Perm);
performanceTrain_raw_Perm{1, 4} = [num2str(performanceTrain_raw_Perm{1,2}, '%.2f'), ' ± ', num2str(performanceTrain_raw_Perm{1,3}, '%.2f')];

performanceTest_raw_Perm{1,  1} = repNumber;
performanceTest_raw_Perm{1,  2} = mean(foldwiseTestAcc_raw_Perm);
performanceTest_raw_Perm{1,  3} = std(foldwiseTestAcc_raw_Perm);
performanceTest_raw_Perm{1,  4} = [num2str(performanceTest_raw_Perm{1, 2}, '%.2f'), ' ± ', num2str(performanceTest_raw_Perm{1, 3}, '%.2f')];

% Model performance: average over folds - harmonized, permutation
performanceTrain_harm_Perm{1, 1} = repNumber;
performanceTrain_harm_Perm{1, 2} = mean(foldwiseTrainAcc_harm_Perm);
performanceTrain_harm_Perm{1, 3} = std(foldwiseTrainAcc_harm_Perm);
performanceTrain_harm_Perm{1, 4} = [num2str(performanceTrain_harm_Perm{1,2}, '%.2f'), ' ± ', num2str(performanceTrain_harm_Perm{1,3}, '%.2f')];

performanceTest_harm_Perm{1,  1} = repNumber;
performanceTest_harm_Perm{1,  2} = mean(foldwiseTestAcc_harm_Perm);
performanceTest_harm_Perm{1,  3} = std(foldwiseTestAcc_harm_Perm);
performanceTest_harm_Perm{1,  4} = [num2str(performanceTest_harm_Perm{1, 2}, '%.2f'), ' ± ', num2str(performanceTest_harm_Perm{1, 3}, '%.2f')];
% -------------------------------------------------------------------------
