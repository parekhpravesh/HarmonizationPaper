function exp003_LearningCurves(dataTable,    regressTIV,   regressAge,   ...
                               regressSex,   allCurves,    numCV,        ...
                               numRepCurves, numPermsLC,   Seeds,        ...
                               toHold,       outDir)
% Function to draw learning curves using a linear SVM classifier
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
% allCurves:    vector of sample sizes per site to be used for drawing
%               learning curves
% numCV:        number of cross-validation splits
% numPerms:     number of times permutation testing should be done
% numRepCurves: number of times the learning curves should be generated
% numPermsLC:   number of times permutation testing of learning curves
%               should be done
% Seeds:        [numPermsLC x 1] vector of seeds to be used during
%               cross-validation and permutation testing
% toHold:       number of samples per site to use for neuroharmonize learn
%               (on which learning curve sample sizes will be applied)
% outDir:       full path to where results should be saved
% 
%% Output:
% A 'Results.mat' file is saved in the outDir containing the results
% 
%% Notes:
% The goal of this experiment is to draw a series of learning curves; for a
% given pool of data, we split the data into k folds - SVMTest set and the
% remaining data; the remaining data is then split into SVMTrain set and
% the NHLearn set. Within the NHLearn set, we sub-select different sample
% sizes (as specified in allCurves) and use them to learn harmonization
% parameters. These harmonization parameters are then applied to SVMTrain
% and SVMTest sets. After this a linear SVM classifier is trained using the
% features in SVMTrain and then predictions are made on SVMTest.
% Permutation testing is performed to test if the classifier performance is
% above chance level
%  
%% Defaults:
% regressTIV:   true
% regressAge:   true
% regressSex:   true
% allCurves:    10:10:200
% numCV:        10
% numRepCurves: 50
% numPermsLC:   100
% Seeds:        rng(0, 'twister'); randi(9999999, numPermsLC, 1)
% toHold:       200
% outDir:       pwd/exp003
% 
%% Authors:
% Bhalerao, Gaurav
% Parekh, Pravesh
% October 30, 2021
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
    regressTIV = false;
else
    if ~islogical(regressTIV)
        error('Expect regressTIV to be either true or false');
    end
end

% Check regressAge
if ~exist('regressAge', 'var') || isempty(regressAge)
    regressAge = false;
else
    if ~islogical(regressAge)
        error('Expect regressAge to be either true or false');
    end
end

% Check regressSex
if ~exist('regressSex', 'var') || isempty(regressSex)
    regressSex = false;
else
    if ~islogical(regressSex)
        error('Expect regressSex to be either true or false');
    end
end

% Check allCurves
if ~exist('allCurves', 'var') || isempty(allCurves)
    allCurves = 10:10:200;
end

% Check numCV
if ~exist('numCV', 'var') || isempty(numCV)
    numCV = 10;
end

% Check numRepCurves
if ~exist('numRepCurves', 'var') || isempty(numRepCurves)
    numRepCurves = 50;
end

% Check numPermsLC
if ~exist('numPermsLC', 'var') || isempty(numPermsLC)
    numPermsLC = 100;
end

% Check Seeds
if ~exist('Seeds', 'var') || isempty(Seeds)
    rng(0, 'twister');
    Seeds = randi(9999999, numPermsLC, 1);
else
    if length(Seeds) ~= numPermsLC
        error(['Expected ', num2str(numPermsLC), ' many seeds']);
    end
end

% Check toHold
if ~exist('toHold', 'var') || isempty(toHold)
    toHold = 200;
end

% Check outDir
if ~exist('outDir', 'var') || isempty(outDir)
    outDir = fullfile(pwd, 'exp008');
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
numSites    = length(unique(dataTable.Site));
numCurves   = length(allCurves);
varNamesLC  = {'RepeatNumber'; 'SampSize_NH_Train'; 'SampSize_SVM_Train'; 'SampSize_SVM_Test'; ...
               'SampleSizePerSite'; 'MeanAccuracy'; 'SD_Accuracy'; 'MeanSD_Accuracy'};

[groundTruth_SVM_Train,         groundTruth_SVM_Test,       ...
 groundTruth_NH_Train_toUse,    predictions_SVM_Train,      ...
 predictions_SVM_Test,          predictions_NH_Train_toUse] = deal(cell(numCV, numCurves, numRepCurves));

[groundTruth_SVM_Perm_Train,    groundTruth_SVM_Perm_Test, ...
 predictions_SVM_Perm_Train,    predictions_SVM_Perm_Test]  = deal(cell(numCV, numCurves, numPermsLC));

[sampSize_SVM_Train,            sampSize_SVM_Test,          ...
 sampSize_NH_Train,             fractionData_NH_Train]      = deal(zeros(numCV, numCurves, numPermsLC));

[performanceTrain,              performanceTest]            = deal(cell(numCurves*numRepCurves, 8));

[performanceTrain_Perm,         performanceTest_Perm]       = deal(cell(numCurves*numPermsLC, 8));

%% LC pipeline: 1 to numRepCurves
count       = 1;
count_res   = 1;
skipCV      = false;
for repeats = 1:numRepCurves
    
    % Start timer
    t_init = tic;
    
    % Locations to save performance data
    loc_init = count_res;
    loc_end  = count_res + numCurves - 1;
    
    [groundTruth_SVM_Train(:, :, count),            groundTruth_SVM_Test(:, :, count),          ...
     groundTruth_SVM_Perm_Train(:, :, count),       groundTruth_SVM_Perm_Test(:, :, count),     ...
     groundTruth_NH_Train_toUse(:, :, count),       predictions_SVM_Train(:, :, count),         ...
     predictions_SVM_Test(:, :, count),             predictions_NH_Train_toUse(:, :, count),    ...
     predictions_SVM_Perm_Train(:, :, count),       predictions_SVM_Perm_Test(:, :, count),     ...
     performanceTrain(loc_init:loc_end, :),         performanceTest(loc_init:loc_end, :),       ...
     performanceTrain_Perm(loc_init:loc_end, :),    performanceTest_Perm(loc_init:loc_end, :),  ...
     sampSize_SVM_Train(:, :, count),               sampSize_SVM_Test(:, :, count),             ...
     sampSize_NH_Train(:, :, count),                fractionData_NH_Train(:, :, count)]    =    ...
     doLCCurves(numCV, Seeds(count),    dataTable,  numSites,   regressTIV,                     ...
               regressAge,  regressSex, multiClass, numCurves,                                  ...
               allCurves,   count,      skipCV,     toHold, outDir);    
    
     % Stop timer
     t_end = toc(t_init);
         
    % Update user
    disp(['Finished Repeat: ', num2str(count, '%02d'), ' [', num2str(t_end, '%.2f'), ' seconds]']);
          
     % Update counter
     count      = count + 1;
     count_res  = count_res + numCurves;
end

%% ML pipeline: numRepeats+1 to numPerms
skipCV      = true;
for repeats = count:numPermsLC
    
    % Start timer
    t_init = tic;
    
    % Locations to save performance data
    loc_init = count_res;
    loc_end  = count_res + numCurves - 1;
    
    [~,                                          ~,                                          ...
     groundTruth_SVM_Perm_Train(:, :, count),    groundTruth_SVM_Perm_Test(:, :, count),     ...
     ~,                                          ~,                                          ...
     ~,                                          ~,                                          ...
     predictions_SVM_Perm_Train(:, :, count),    predictions_SVM_Perm_Test(:, :, count),     ...
     ~,                                          ~,                                          ...
     performanceTrain_Perm(loc_init:loc_end, :), performanceTest_Perm(loc_init:loc_end, :),  ...
     sampSize_SVM_Train(:, :, count),            sampSize_SVM_Test(:, :, count),             ...
     sampSize_NH_Train(:, :, count),             fractionData_NH_Train(:, :, count)]    =    ...
     doLCCurves(numCV, Seeds(count),    dataTable,  numSites,   regressTIV,                  ...
               regressAge,  regressSex, multiClass, numCurves,                               ...
               allCurves,   count,      skipCV,     toHold, outDir);
           
     % Stop timer
     t_end = toc(t_init);
         
    % Update user
    disp(['Finished Repeat: ', num2str(count, '%02d'), ' [', num2str(t_end, '%.2f'), ' seconds]']);          
          
     % Update counter
     count      = count + 1;
     count_res  = count_res + numCurves;
end

%% Convert to tables
performanceTrain       = cell2table(performanceTrain,       'VariableNames', varNamesLC); %#ok<NASGU>
performanceTest        = cell2table(performanceTest,        'VariableNames', varNamesLC);
performanceTrain_Perm  = cell2table(performanceTrain_Perm,  'VariableNames', varNamesLC); %#ok<NASGU>
performanceTest_Perm   = cell2table(performanceTest_Perm,   'VariableNames', varNamesLC);

%% Compute p values
% For each original accuracy, each sample size, count the number of times 
% the permutation accuracy became equal to or exceeded this value
% Alternatively, error is lesser in permutation set

% Initialize
overallpValue_test = zeros(numCurves, 1);

for curves = 1:numCurves
    
    % Initialize for this curve
    pValues_test = zeros(numRepCurves,1);
    
    % Go over each repeat
    for repeat = 1:numRepCurves
        
        % Subset variables - actual accuracies
        loc_act      = performanceTest.RepeatNumber == repeat & performanceTest.SampleSizePerSite == allCurves(curves);
        act_accuracy = performanceTest.MeanAccuracy(loc_act);
        
        % Subset variables - permutation accuracies
        perm_accuracy = performanceTest_Perm.MeanAccuracy(performanceTest_Perm.SampleSizePerSite == allCurves(curves));
        
        % Calculate p values
        pValues_test(repeat, 1) = (sum(perm_accuracy >= act_accuracy) + 1)/(numPermsLC + 1);
        
        % Append p value to that repeat and sample size per site
        performanceTest.pValueType(loc_act) = pValues_test(repeat);
    end

% Average p values
overallpValue_test(curves,1) = mean(pValues_test);
end

%% Save everything
save(fullfile(outDir, 'Results.mat'));

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

function [hTrainData, hremTrainData, hTestData] = doHarmonization_LC(trainData, remTrainData, testData, ...
                                                                     trainSite, remTrainSite, testSite, ...
                                                                     trainCov,  remTrainCov,  testCov,  ...
                                                                     covNames,  tmpDir, foldNum)
% Function to call neuroHarmonize and return harmonized data

% Fold number
foldStr = ['fold_', num2str(foldNum, '%02d')];

% Write out csv files
dlmwrite(fullfile(tmpDir, [foldStr, '_temp_trainData.csv']),    trainData);
dlmwrite(fullfile(tmpDir, [foldStr, '_temp_remTrainData.csv']), remTrainData);
dlmwrite(fullfile(tmpDir, [foldStr, '_temp_testData.csv']),     testData);

% Put covariates together
trainSite    = [trainSite,    num2cell(trainCov)];
remTrainSite = [remTrainSite, num2cell(remTrainCov)];
testSite     = [testSite,     num2cell(testCov)];

% Create covariates files for writing
fid_train    = fopen(fullfile(tmpDir, [foldStr, '_temp_trainSite.csv']),    'w');
fid_remTrain = fopen(fullfile(tmpDir, [foldStr, '_temp_remTrainSite.csv']), 'w');
fid_test     = fopen(fullfile(tmpDir, [foldStr, '_temp_testSite.csv']),     'w');

% Print header
% Walter's example: https://www.mathworks.com/matlabcentral/answers/364295
tmpHeader = ['SITE', covNames];   
fprintf(fid_train,      '%s,',  tmpHeader{1:end-1});
fprintf(fid_train,      '%s\n', tmpHeader{end});
fprintf(fid_remTrain,   '%s,',  tmpHeader{1:end-1});
fprintf(fid_remTrain,   '%s\n', tmpHeader{end});
fprintf(fid_test,       '%s,',  tmpHeader{1:end-1});
fprintf(fid_test,       '%s\n', tmpHeader{end});

% Write out data: integer or float doesn't seem to matter
for lines = 1:size(trainSite,1)
    fprintf(fid_train, '%s,%f,%f,%f\n', trainSite{lines,:});
end
for lines = 1:size(remTrainSite,1)
    fprintf(fid_remTrain, '%s,%f,%f,%f\n', remTrainSite{lines,:});
end
for lines = 1:size(testSite,1)
    fprintf(fid_test, '%s,%f,%f,%f\n', testSite{lines,:});
end

% Close files
fclose(fid_train);
fclose(fid_remTrain);
fclose(fid_test);

% Convert covariates to tables
% trainSite    = cell2table(trainSite,    'VariableNames', ['SITE', covNames]);
% remTrainSite = cell2table(remTrainSite, 'VariableNames', ['SITE', covNames]);
% testSite     = cell2table(testSite,     'VariableNames', ['SITE', covNames]);

% Write out covariates files
% writetable(trainSite,    fullfile(tmpDir, [foldStr, '_temp_trainSite.csv']));
% writetable(remTrainSite, fullfile(tmpDir, [foldStr, '_temp_remTrainSite.csv']));
% writetable(testSite,     fullfile(tmpDir, [foldStr, '_temp_testSite.csv']));

% Might need to add full path to doHarmonization_LC.py script
command = ['python doHarmonization_LC.py ', tmpDir, ' ', foldStr];
system(command);

% Read adjusted data back in
hTrainData      = dlmread(fullfile(tmpDir, [foldStr, '_adjustedTrainData.csv']));
hremTrainData   = dlmread(fullfile(tmpDir, [foldStr, '_adjustedremTrainData.csv']));
hTestData       = dlmread(fullfile(tmpDir, [foldStr, '_adjustedTestData.csv']));

% Delete files
delete(fullfile(tmpDir, [foldStr, '_temp_trainData.csv']));
delete(fullfile(tmpDir, [foldStr, '_temp_remTrainData.csv']));
delete(fullfile(tmpDir, [foldStr, '_temp_testData.csv']));

delete(fullfile(tmpDir, [foldStr, '_temp_trainSite.csv']));
delete(fullfile(tmpDir, [foldStr, '_temp_remTrainSite.csv']));
delete(fullfile(tmpDir, [foldStr, '_temp_testSite.csv']));

delete(fullfile(tmpDir, [foldStr, '_adjustedTrainData.csv']));
delete(fullfile(tmpDir, [foldStr, '_adjustedremTrainData.csv']));
delete(fullfile(tmpDir, [foldStr, '_adjustedTestData.csv']));
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
          
function ...
[groundTruth_SVM_Train,       groundTruth_SVM_Test,         ...
 groundTruth_SVM_Perm_Train,  groundTruth_SVM_Perm_Test,    ...
 groundTruth_NH_Train_toUse,  predictions_SVM_Train,        ...
 predictions_SVM_Test,        predictions_NH_Train_toUse,   ...
 predictions_SVM_Perm_Train,  predictions_SVM_Perm_Test,    ...
 performanceTrain,            performanceTest,              ...
 performanceTrain_Perm,       performanceTest_Perm,         ...
 sampSize_SVM_Train,          sampSize_SVM_Test,            ...
 sampSize_NH_Train,           fractionData_NH_Train]    =   ...
doLCCurves(numCV, seed, dataTable,  numSites,   regressTIV, ...
           regressAge,  regressSex, multiClass, numCurves,  ...
           sampleSizesPerSite, repNumber, skipCV, toHold, outDir)

% This function prepares data for creating learning curves and then runs
% all folds and curves

% Set seed
rng(seed, 'twister');

% Make a working version of dataTable
workTable = dataTable;

% Generate a k-fold partition
cv = cvpartition(workTable.Site, 'KFold', numCV, 'stratify', true);

% Names of covariates
covNames = {'TIV', 'Age', 'Sex'};

% Initialization
[sampSize_SVM_Train,          sampSize_SVM_Test,          ...
 sampSize_NH_Train,           fractionData_NH_Train]      = deal(zeros(numCV, numCurves));

[groundTruth_SVM_Train,       groundTruth_SVM_Test,       ...
 groundTruth_SVM_Perm_Train,  groundTruth_SVM_Perm_Test,  ...
 groundTruth_NH_Train_toUse,  predictions_SVM_Train,      ...
 predictions_SVM_Test,        predictions_NH_Train_toUse, ...
 predictions_SVM_Perm_Train,  predictions_SVM_Perm_Test]  = deal(cell(numCV, numCurves));

[accuracy_SVM_Train,          accuracy_SVM_Test,          ...
 accuracy_NH_Train_toUse,     accuracy_SVM_Perm_Train,    ...
 accuracy_SVM_Perm_Test]                                  = deal(NaN(numCV, numCurves));

[performanceTrain,            performanceTest,            ...
 performanceTrain_Perm,       performanceTest_Perm]       = deal(cell(numCurves, 8));

% Go over folds
for fold = 1:numCV
    
    % Test data for this fold is the test data for SVM - constant across
    % all learning curves within this fold
    dataTable_SVM_Test = workTable(cv.test(fold),    :);
    
    % Training data for this fold will be split into two parts: one for
    % learing neuroharmonize and one will later be used as training data 
    % for SVM
    dataAll_NH = workTable(cv.training(fold),:);
    
    % Perform a split on this data
    cv_hold = cvpartition(dataAll_NH.Site, 'Holdout', numSites*toHold, 'Stratify', true);
    
    % Further subset data: the training data from this will be sub-sampled
    % and then used to learn harmonization parameters; the learned 
    % parameters from harmonization will be applied to the test data; then,
    % the harmonized test data will be used to train a SVM classifier
    dataTable_NH_Train = dataAll_NH(cv_hold.test,     :);
    dataTable_NH_Test  = dataAll_NH(cv_hold.training, :);
    
    % Go over various learning curve sizes
    for curves = 1:numCurves
        
        % Special case where last learning curve is being executed; since
        % the last entry of allCurves is basically the entire
        % dataTable_NH_Train to be used for neuroHarmonize trianing, we do
        % not create the intCV object
        if curves == numCurves
            dataTable_NH_Train_toUse            = dataTable_NH_Train;
            fractionData_NH_Train(fold, curves) = 100;
        else
            % Calculate fraction of training data to be uesd for learning
            % harmonization
            estPercentage = (sampleSizesPerSite(curves)*numSites)/height(dataTable_NH_Train);

            % Split training data into the necessary sample size
            intCV = cvpartition(dataTable_NH_Train.Site, 'HoldOut', estPercentage, 'stratify', true);

            % Subset data
            dataTable_NH_Train_toUse = dataTable_NH_Train(intCV.test, :);
            
            % Save estimated percentage
            fractionData_NH_Train(fold, curves)  = estPercentage;
        end
        
        % Working copy of training and test data for SVM
        dataTable_NH_Test_toUse  = dataTable_NH_Test;
        dataTable_SVM_Test_toUse = dataTable_SVM_Test;

        % Record all sample sizes at this stage: (fold, curve)
        sampSize_SVM_Train(fold, curves)     = height(dataTable_NH_Test_toUse);
        sampSize_SVM_Test(fold,  curves)     = height(dataTable_SVM_Test_toUse);
        sampSize_NH_Train(fold,  curves)     = height(dataTable_NH_Train_toUse);
        
        % Prepare features, covariates, and site information for neuroHarmonize
        features_NH_Train = dataTable_NH_Train_toUse{:, 8:end};
        features_NH_Test  = dataTable_NH_Test_toUse{:,  8:end};
        features_NH_Ext   = dataTable_SVM_Test_toUse{:, 8:end};

        covars_NH_Train   = [dataTable_NH_Train_toUse.TIV, dataTable_NH_Train_toUse.Age, dataTable_NH_Train_toUse.Female];
        covars_NH_Test    = [dataTable_NH_Test_toUse.TIV,  dataTable_NH_Test_toUse.Age,  dataTable_NH_Test_toUse.Female];
        covars_NH_Ext     = [dataTable_SVM_Test_toUse.TIV, dataTable_SVM_Test_toUse.Age, dataTable_SVM_Test_toUse.Female];

        site_NH_Train     = dataTable_NH_Train_toUse.Site;
        site_NH_Test      = dataTable_NH_Test_toUse.Site;
        site_NH_Ext       = dataTable_SVM_Test_toUse.Site;
        
        % Send to neuroHarmonize
        [features_NH_Train, features_NH_Ext, features_NH_Test] = doHarmonization_LC(features_NH_Train, features_NH_Ext, features_NH_Test,   ...
                                                                                    site_NH_Train,     site_NH_Ext,     site_NH_Test,       ...
                                                                                    covars_NH_Train,   covars_NH_Ext,   covars_NH_Test,     ...
                                                                                    covNames,          outDir,          fold);
                                                                                
        % Additional safety check; if harmonized features have negative
        % values, it means that neuroHarmonize did not work correctly;
        % warn user and then discard this sample size
        if sum(features_NH_Train    < 0, 'all')  > 0 || ...
           sum(features_NH_Ext      < 0, 'all')  > 0 || ...
           sum(features_NH_Test     < 0, 'all')  > 0
            warning(['Negative values found at: ', num2str(sampleSizesPerSite(curves)), ' samples per site; aborting']);
            
            % Set dummy values
            groundTruth_SVM_Train{fold, curves}       = cell(length(site_NH_Test),  1);
            groundTruth_SVM_Test{fold, curves}        = cell(length(site_NH_Ext),   1);
            groundTruth_SVM_Perm_Train{fold, curves}  = cell(length(site_NH_Test),  1);
            groundTruth_SVM_Perm_Test{fold, curves}   = cell(length(site_NH_Ext),   1);
            groundTruth_NH_Train_toUse{fold, curves}  = cell(length(site_NH_Train), 1);
            
            predictions_SVM_Train{fold, curves}       = cell(length(site_NH_Test),  1);
            predictions_SVM_Test{fold, curves}        = cell(length(site_NH_Ext),   1);
            predictions_NH_Train_toUse{fold, curves}  = cell(length(site_NH_Train), 1);
            predictions_SVM_Perm_Train{fold, curves}  = cell(length(site_NH_Test),  1);
            predictions_SVM_Perm_Test{fold, curves}   = cell(length(site_NH_Ext),   1);
            
            saveName = fullfile(outDir, ['NegValues_', num2str(sampleSizesPerSite(curves), '%02d'), 'SamplesPerSite_', num2str(fold, '%02d'), 'Fold_', num2str(curves, '%02d'), 'Curve.mat']);
            save(saveName);
            continue;
        end
        
        % Update features in data tables
        dataTable_NH_Train_toUse{:, 8:end}  = features_NH_Train;
        dataTable_NH_Test_toUse{:,  8:end}  = features_NH_Test;
        dataTable_SVM_Test_toUse{:, 8:end}  = features_NH_Ext;
        
        % Prepare features, covariates, and site information for SVM
        features_NH_Train_toUse = dataTable_NH_Train_toUse{:, 8:end};
        features_SVM_Train      = dataTable_NH_Test_toUse{:, 8:end};
        features_SVM_Test       = dataTable_SVM_Test_toUse{:, 8:end};

        labels_NH_Train_toUse   = dataTable_NH_Train_toUse.Site;
        labels_SVM_Train        = dataTable_NH_Test_toUse.Site;
        labels_SVM_Test         = dataTable_SVM_Test_toUse.Site;

        covars_NH_Train_toUse   = [dataTable_NH_Train_toUse.TIV, dataTable_NH_Train_toUse.Age,  dataTable_NH_Train_toUse.Female];
        covars_SVM_Train        = [dataTable_NH_Test_toUse.TIV,  dataTable_NH_Test_toUse.Age,   dataTable_NH_Test_toUse.Female];
        covars_SVM_Test         = [dataTable_SVM_Test_toUse.TIV, dataTable_SVM_Test_toUse.Age,  dataTable_SVM_Test_toUse.Female];

        % Regress covariates
        toRegress             = [regressTIV, regressAge, regressSex];
        covars_NH_Train_toUse = covars_NH_Train_toUse(:, toRegress);
        covars_SVM_Train      = covars_SVM_Train(:, toRegress);
        covars_SVM_Test       = covars_SVM_Test(:,  toRegress);

        if ~isempty(covars_SVM_Train)
            [features_SVM_Train, coeff]  = regressVariables(features_SVM_Train,      covars_SVM_Train);
            features_SVM_Test            = regressVariables(features_SVM_Test,       covars_SVM_Test,       coeff);
            features_NH_Train_toUse      = regressVariables(features_NH_Train_toUse, covars_NH_Train_toUse, coeff);
        end

        % Standardize features
        [features_SVM_Train, coeff_std] = standardizeData(features_SVM_Train);
         features_SVM_Test              = standardizeData(features_SVM_Test,        coeff_std);
         features_NH_Train_toUse        = standardizeData(features_NH_Train_toUse,  coeff_std);
         
        % Prepare permutation labels
        tmp_labelsAll           = [labels_SVM_Train; labels_SVM_Test];
        numSamples              = length(tmp_labelsAll);
        randOrder               = randperm(numSamples, numSamples);
        tmp_labelsAll           = tmp_labelsAll(randOrder);
        labels_SVM_Perm_Train   = tmp_labelsAll(1:length(labels_SVM_Train));
        labels_SVM_Perm_Test    = tmp_labelsAll(length(labels_SVM_Train)+1:end);
        
        % Train classifier
        if multiClass
            if ~skipCV
                mdl_SVM     = fitcecoc(features_SVM_Train, labels_SVM_Train,        'Coding', 'onevsone', 'Learners', 'svm');
            end
            mdl_SVM_Perm    = fitcecoc(features_SVM_Train, labels_SVM_Perm_Train,   'Coding', 'onevsone', 'Learners', 'svm');
        else
            if ~skipCV
                mdl_SVM     = fitcsvm(features_SVM_Train,  labels_SVM_Train,        'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
            end
            mdl_SVM_Perm    = fitcsvm(features_SVM_Train,  labels_SVM_Perm_Train,   'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
        end
        
        % Record ground truth for posterity
        groundTruth_SVM_Train{fold, curves}       = labels_SVM_Train;
        groundTruth_SVM_Test{fold, curves}        = labels_SVM_Test;
        groundTruth_SVM_Perm_Train{fold, curves}  = labels_SVM_Perm_Train;
        groundTruth_SVM_Perm_Test{fold, curves}   = labels_SVM_Perm_Test;
        groundTruth_NH_Train_toUse{fold, curves}  = labels_NH_Train_toUse;
        
        % Make predictions
        if ~skipCV
            predictions_SVM_Train{fold, curves}      = predict(mdl_SVM, features_SVM_Train);
            predictions_SVM_Test{fold, curves}       = predict(mdl_SVM, features_SVM_Test);
            predictions_NH_Train_toUse{fold, curves} = predict(mdl_SVM, features_NH_Train_toUse);
        else
            predictions_SVM_Train{fold, curves}      = cell(length(labels_SVM_Train),       1);
            predictions_SVM_Test{fold, curves}       = cell(length(labels_SVM_Test),        1);
            predictions_NH_Train_toUse{fold, curves} = cell(length(labels_NH_Train_toUse), 1);
        end
        predictions_SVM_Perm_Train{fold, curves} = predict(mdl_SVM_Perm, features_SVM_Train);
        predictions_SVM_Perm_Test{fold, curves}  = predict(mdl_SVM_Perm, features_SVM_Test);

        % Evaluate classifier performances
        if ~skipCV
            accuracy_SVM_Train(fold, curves)      = sum(strcmpi(predictions_SVM_Train{fold, curves},        labels_SVM_Train))/length(labels_SVM_Train)           * 100;
            accuracy_SVM_Test(fold, curves)       = sum(strcmpi(predictions_SVM_Test{fold, curves},         labels_SVM_Test))/length(labels_SVM_Test)             * 100;
            accuracy_NH_Train_toUse(fold, curves) = sum(strcmpi(predictions_NH_Train_toUse{fold, curves},   labels_NH_Train_toUse))/length(labels_NH_Train_toUse) * 100;
        else
            accuracy_SVM_Train(fold, curves)      = NaN;
            accuracy_SVM_Test(fold, curves)       = NaN;
            accuracy_NH_Train_toUse(fold, curves) = NaN;
        end
        accuracy_SVM_Perm_Train(fold, curves)    = sum(strcmpi(predictions_SVM_Perm_Train{fold, curves},  labels_SVM_Perm_Train))/length(labels_SVM_Perm_Train) * 100;
        accuracy_SVM_Perm_Test(fold, curves)     = sum(strcmpi(predictions_SVM_Perm_Test{fold, curves},   labels_SVM_Perm_Test))/length(labels_SVM_Perm_Test)   * 100;
    end
end

% Summarize results for this fold: repeat number, sample size for
% neuroharmonize learn, sample size for SVM learn, sample size for SVM 
% test, sample size per site as in allCurves, mean accuracy, standard 
% deviation of accuracy, and a text version containing mean and SD

% varNames = {'RepeatNumber'; 'SampSize_NH_Train'; 'SampSize_SVM_Train'; 'SampSize_SVM_Test';  ...
%             'SampleSizePerSite'; 'MeanAccuracy'; 'SD_Accuracy';        'MeanSD_Accuracy'};

for curves = 1:numCurves
    if ~skipCV        
        % Model performance: average over folds - harmonized, actual data
        performanceTrain{curves, 1} = repNumber;
        performanceTrain{curves, 2} = round(mean(sampSize_NH_Train(:, curves)),  0);
        performanceTrain{curves, 3} = round(mean(sampSize_SVM_Train(:, curves)), 0);
        performanceTrain{curves, 4} = round(mean(sampSize_SVM_Test(:, curves)),  0);
        performanceTrain{curves, 5} = sampleSizesPerSite(curves);
        performanceTrain{curves, 6} = mean(accuracy_SVM_Train(:, curves),    'omitnan');
        performanceTrain{curves, 7} = std(accuracy_SVM_Train(:, curves), [], 'omitnan');
        performanceTrain{curves, 8} = [num2str(performanceTrain{curves, 6}, '%.2f'), ' ± ', num2str(performanceTrain{curves, 7}, '%.2f')];
        
        performanceTest{curves, 1}  = repNumber;
        performanceTest{curves, 2}  = round(mean(sampSize_NH_Train(:, curves)),  0);
        performanceTest{curves, 3}  = round(mean(sampSize_SVM_Train(:, curves)), 0);
        performanceTest{curves, 4}  = round(mean(sampSize_SVM_Test(:, curves)),  0);
        performanceTest{curves, 5}  = sampleSizesPerSite(curves);
        performanceTest{curves, 6}  = mean(accuracy_SVM_Test(:, curves),    'omitnan');
        performanceTest{curves, 7}  = std(accuracy_SVM_Test(:, curves), [], 'omitnan');
        performanceTest{curves, 8}  = [num2str(performanceTest{curves, 6}, '%.2f'), ' ± ', num2str(performanceTest{curves, 7}, '%.2f')];
    else
        % Set these to NaN
        performanceTrain{curves, 1} = NaN;
        performanceTrain{curves, 2} = NaN;
        performanceTrain{curves, 3} = NaN;
        performanceTrain{curves, 4} = NaN;
        performanceTrain{curves, 5} = NaN;        
        performanceTrain{curves, 6} = NaN;
        performanceTrain{curves, 7} = NaN;
        performanceTrain{curves, 8} = NaN;
        
        performanceTest{curves, 1}  = NaN;
        performanceTest{curves, 2}  = NaN;
        performanceTest{curves, 3}  = NaN;
        performanceTest{curves, 4}  = NaN;
        performanceTest{curves, 5}  = NaN;
        performanceTest{curves, 6}  = NaN;
        performanceTest{curves, 7}  = NaN;
        performanceTest{curves, 8}  = NaN;
    end

    % Model performance: average over folds - harmonized, permutation
    performanceTrain_Perm{curves, 1} = repNumber;
    performanceTrain_Perm{curves, 2} = round(mean(sampSize_NH_Train(:, curves)),  0);
    performanceTrain_Perm{curves, 3} = round(mean(sampSize_SVM_Train(:, curves)), 0);
    performanceTrain_Perm{curves, 4} = round(mean(sampSize_SVM_Test(:, curves)),  0);
    performanceTrain_Perm{curves, 5} = sampleSizesPerSite(curves);
    performanceTrain_Perm{curves, 6} = mean(accuracy_SVM_Perm_Train(:, curves),    'omitnan');
    performanceTrain_Perm{curves, 7} = std(accuracy_SVM_Perm_Train(:, curves), [], 'omitnan');
    performanceTrain_Perm{curves, 8} = [num2str(performanceTrain_Perm{curves, 6}, '%.2f'), ' ± ', num2str(performanceTrain_Perm{curves, 7}, '%.2f')];
    
    performanceTest_Perm{curves, 1}  = repNumber;
    performanceTest_Perm{curves, 2}  = round(mean(sampSize_NH_Train(:, curves)),  0);
    performanceTest_Perm{curves, 3}  = round(mean(sampSize_SVM_Train(:, curves)), 0);
    performanceTest_Perm{curves, 4}  = round(mean(sampSize_SVM_Test(:, curves)),  0);
    performanceTest_Perm{curves, 5}  = sampleSizesPerSite(curves);    
    performanceTest_Perm{curves, 6}  = mean(accuracy_SVM_Perm_Test(:, curves),    'omitnan');
    performanceTest_Perm{curves, 7}  = std(accuracy_SVM_Perm_Test(:, curves), [], 'omitnan');
    performanceTest_Perm{curves, 8}  = [num2str(performanceTest_Perm{curves, 6}, '%.2f'), ' ± ', num2str(performanceTest_Perm{curves, 7}, '%.2f')];    
end
% -------------------------------------------------------------------------