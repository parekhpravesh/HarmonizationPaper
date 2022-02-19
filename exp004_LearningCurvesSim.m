function exp004_LearningCurvesSim(dataTable,  allCurves, numRepCurves, ...
                                  numPermsLC, Seeds,     outDir)
% Function to draw learning curves for a linear SVM classification given a 
% set of parameters
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
% allCurves:    vector of sample sizes per site to be used for drawing
%               learning curves
% numPerms:     number of times permutation testing should be done
% numRepCurves: number of times the learning curves should be generated
% numPermsLC:   number of times permutation testing of learning curves
%               should be done
% Seeds:        [numPermsLC x 1] vector of seeds to be used during
%               cross-validation and permutation testing
% outDir:       full path to where results should be saved
% 
%% Output:
% A 'Results.mat' file is saved in the outDir containing the results
% 
%% Notes:
% The goal of this experiment is to draw a series of learning curves using 
% simulated data; for a given pool of data, we split the data into 20 folds 
% - SVMTest set and the remaining data; the remaining data is then split 
% into SVMTrain set (70 samples per site) and the NHLearn set. Within the 
% NHLearn set, we sub-select different sample sizes (as specified in 
% allCurves) and use them to learn harmonization parameters. These 
% harmonization parameters are then applied to SVMTrain and SVMTest sets. 
% After this a linear SVM classifier is trained using the features in 
% SVMTrain and then predictions are made on SVMTest. Permutation testing 
% is performed to test if the classifier performance is above chance level
% 
%% Defaults:
% allCurves:    10:10:500
% numRepCurves: 50
% numPermsLC:   100
% Seeds:        rng(0, 'twister'); randi(9999999, numPermsLC, 1)
% outDir:       pwd/exp004
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

% Check allCurves
if ~exist('allCurves', 'var') || isempty(allCurves)
    allCurves = 10:10:500;
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

% Check outDir
if ~exist('outDir', 'var') || isempty(outDir)
    outDir = fullfile(pwd, 'exp004');
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
numSites  = length(unique(dataTable.Site));
numCurves = length(allCurves);

[groundTruth_SVM_Train,         groundTruth_SVM_Test,       ...
 predictions_SVM_Train,         predictions_SVM_Test,       ...
 groundTruth_SVM_Perm_Train,    groundTruth_SVM_Perm_Test,  ...
 predictions_SVM_Perm_Train,    predictions_SVM_Perm_Test,  ...
 accuracy_SVM_Train,            accuracy_SVM_Test,          ...
 accuracy_SVM_Perm_Train,       accuracy_SVM_Perm_Test,     ...
 pValues_test,                  overall_pValue,             ...
 MD_noHarm_SVMTrain,            MD_noHarm_SVMTest,          ...
 MD_noHarm_NHLearn,             MD_Harm_SVMTrain,           ...
 MD_Harm_SVMTest,               MD_Harm_NHLearn] = deal(cell(numCurves, 1));

%% Run experiments
for curves = 1:numCurves
    
    % Start timer
    t_init = tic;

    % Run numRepCurves and numPermsLC
    [groundTruth_SVM_Train{curves,1},        groundTruth_SVM_Test{curves,1},       ...
     predictions_SVM_Train{curves,1},        predictions_SVM_Test{curves,1},       ...
     groundTruth_SVM_Perm_Train{curves,1},   groundTruth_SVM_Perm_Test{curves,1},  ...
     predictions_SVM_Perm_Train{curves,1},   predictions_SVM_Perm_Test{curves,1},  ...
     accuracy_SVM_Train{curves,1},           accuracy_SVM_Test{curves,1},          ...
     accuracy_SVM_Perm_Train{curves,1},      accuracy_SVM_Perm_Test{curves,1},     ...
     pValues_test{curves,1},                 overall_pValue{curves,1},             ...
     MD_noHarm_SVMTrain{curves,1},           MD_noHarm_SVMTest{curves,1},          ...
     MD_noHarm_NHLearn{curves,1},            MD_Harm_SVMTrain{curves,1},           ...
     MD_Harm_SVMTest{curves,1},              MD_Harm_NHLearn{curves,1}]         =  ...
     doLC(dataTable, numSites, numRepCurves, numPermsLC, allCurves(curves), Seeds, multiClass, outDir);
 
     % Stop timer
     t_end = toc(t_init);
 
    % Evaluate p value
    if overall_pValue{curves,1} < 0.05
        disp(['Finished sample size : ', num2str(allCurves(curves)), ' [', num2str(t_end, '%.2f'), ' seconds]']);
    else
        % Compile results table
        results = cell2table([num2cell(allCurves'), overall_pValue], 'VariableNames', {'SampleSizePerSite'; 'Overall_pValue'}); %#ok<NASGU>
        disp(['Completed sample size : ', num2str(allCurves(curves)), ' [', num2str(t_end, '%.2f'), ' seconds]']);
        break
    end
end

% Save everything
save(fullfile(outDir, 'Results.mat'), 'dataTable', 'curves', 'allCurves', 'pValues*', 'overall*');
try
    save(fullfile(outDir, 'AdditionalResults.mat'), '-v7.3');
catch
end

function [hTrainData, hremTrainData, hTestData] = doHarmonization_LC(trainData, remTrainData, testData, ...
                                                                     trainSite, remTrainSite, testSite, tmpDir)
% Function to call neuroHarmonize and return harmonized data

% Write out csv files
dlmwrite(fullfile(tmpDir, 'temp_trainData.csv'),    trainData);
dlmwrite(fullfile(tmpDir, 'temp_remTrainData.csv'), remTrainData);
dlmwrite(fullfile(tmpDir, 'temp_testData.csv'),     testData);

% Create covariates files for writing
fid_train    = fopen(fullfile(tmpDir, 'temp_trainSite.csv'),    'w');
fid_remTrain = fopen(fullfile(tmpDir, 'temp_remTrainSite.csv'), 'w');
fid_test     = fopen(fullfile(tmpDir, 'temp_testSite.csv'),     'w');

% Print header
% Walter's example: https://www.mathworks.com/matlabcentral/answers/364295
tmpHeader = 'SITE';
fprintf(fid_train,      '%s\n',  tmpHeader);
fprintf(fid_remTrain,   '%s\n',  tmpHeader);
fprintf(fid_test,       '%s\n',  tmpHeader);

% Write out data: integer or float doesn't seem to matter
for lines = 1:size(trainSite,1)
    fprintf(fid_train, '%s\n', trainSite{lines,:});
end
for lines = 1:size(remTrainSite,1)
    fprintf(fid_remTrain, '%s\n', remTrainSite{lines,:});
end
for lines = 1:size(testSite,1)
    fprintf(fid_test, '%s\n', testSite{lines,:});
end

% Close files
fclose(fid_train);
fclose(fid_remTrain);
fclose(fid_test);

% Might need to add full path to doHarmonization_LCSim.py script
command = ['python doHarmonization_LCSim.py ', tmpDir];
system(command);

% Read adjusted data back in
hTrainData      = dlmread(fullfile(tmpDir, 'adjustedTrainData.csv'));
hremTrainData   = dlmread(fullfile(tmpDir, 'adjustedremTrainData.csv'));
hTestData       = dlmread(fullfile(tmpDir, 'adjustedTestData.csv'));

% Delete files
delete(fullfile(tmpDir, 'temp_trainData.csv'));
delete(fullfile(tmpDir, 'temp_remTrainData.csv'));
delete(fullfile(tmpDir, 'temp_testData.csv'));

delete(fullfile(tmpDir, 'temp_trainSite.csv'));
delete(fullfile(tmpDir, 'temp_remTrainSite.csv'));
delete(fullfile(tmpDir, 'temp_testSite.csv'));

delete(fullfile(tmpDir, 'adjustedTrainData.csv'));
delete(fullfile(tmpDir, 'adjustedremTrainData.csv'));
delete(fullfile(tmpDir, 'adjustedTestData.csv'));
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

function [groundTruth_SVM_Train,        groundTruth_SVM_Test,       ...
          predictions_SVM_Train,        predictions_SVM_Test,       ...
          groundTruth_SVM_Perm_Train,   groundTruth_SVM_Perm_Test,  ...
          predictions_SVM_Perm_Train,   predictions_SVM_Perm_Test,  ...
          accuracy_SVM_Train,           accuracy_SVM_Test,          ...
          accuracy_SVM_Perm_Train,      accuracy_SVM_Perm_Test,     ...
          pValues_test,                 overall_pValue,             ...
          MD_noHarm_SVMTrain,           MD_noHarm_SVMTest,          ...
          MD_noHarm_NHLearn,            MD_Harm_SVMTrain,           ...
          MD_Harm_SVMTest,              MD_Harm_NHLearn]         =  ...
          doLC(dataTable, numSites, numRepeats, numPermsLC, currSampleSize, seeds, multiClass, outDir)

% Initialize
[groundTruth_SVM_Train,        groundTruth_SVM_Test,        ...
 predictions_SVM_Train,        predictions_SVM_Test,        ...
 groundTruth_SVM_Perm_Train,   groundTruth_SVM_Perm_Test,   ...
 predictions_SVM_Perm_Train,   predictions_SVM_Perm_Test] = deal(cell(numPermsLC, 1));

[accuracy_SVM_Train,            accuracy_SVM_Test,          ...
 accuracy_SVM_Perm_Train,       accuracy_SVM_Perm_Test,     ...
 MD_noHarm_SVMTrain,            MD_noHarm_SVMTest,          ...
 MD_noHarm_NHLearn,             MD_Harm_SVMTrain,           ...
 MD_Harm_SVMTest,               MD_Harm_NHLearn]      = deal(zeros(numPermsLC, 20));

[accuracy_SVM_Train_repeats,            accuracy_SVM_Test_repeats,          ...
 accuracy_SVM_Perm_Train_repeats,      accuracy_SVM_Perm_Test_repeats]    = deal(zeros(numPermsLC, 1));

allSites = unique(dataTable.Site);

for repeats = 1:numRepeats
    
    % Set seed
    rng(seeds(repeats), 'twister');
    
    % Separate SVM Test samples
    cv = cvpartition(dataTable.Site, 'KFold', 20, 'stratify', true);
    
    for fold = 1:20
    
        % Extract SVM test and remaining data
        data_SVM_Test  = dataTable(cv.test(fold),:);
        data_remaining = dataTable(cv.training(fold),:);

        % Split the remaining data into SVM train and NH learn
        cv2 = cvpartition(data_remaining.Site, 'Holdout', 70*numSites, 'stratify', true);
    
        % Extract SVM train and NH learn
        data_SVM_Train = data_remaining(cv2.test,:);
        data_rem_Learn = data_remaining(cv2.training,:);
        
        if currSampleSize*numSites == height(data_rem_Learn)
            data_NH_Learn = data_rem_Learn;
        else
            % Take currSampleSize from NH Learn
            cv3           = cvpartition(data_rem_Learn.Site, 'HoldOut', currSampleSize*numSites, 'stratify', true);
            data_NH_Learn = data_rem_Learn(cv3.test,:);
        end

        % Extract features
        features_NH_Learn  = data_NH_Learn{:,  8:end};
        features_SVM_Train = data_SVM_Train{:, 8:end};
        features_SVM_Test  = data_SVM_Test{:,  8:end};

        % Extract site information
        site_NH_Learn     = data_NH_Learn.Site;
        site_SVM_Train    = data_SVM_Train.Site;
        site_SVM_Test     = data_SVM_Test.Site;
        
        % Find sites
        for sites = 1:length(allSites)
            loc_SVMTrain(:,sites) = ismember(data_SVM_Train.Site, allSites(sites)); %#ok<AGROW>
            loc_SVMTest(:, sites) = ismember(data_SVM_Test.Site,  allSites(sites)); %#ok<AGROW>
            loc_NHlearn(:, sites) = ismember(data_NH_Learn.Site,  allSites(sites)); %#ok<AGROW>
        end
        
        % Send to neuroHarmonize
        [features_NH_Learn, features_SVM_Train, features_SVM_Test] = doHarmonization_LC(features_NH_Learn, features_SVM_Train, features_SVM_Test,  ...
                                                                                         site_NH_Learn,    site_SVM_Train,     site_SVM_Test,      outDir); %#ok<ASGLU>

        % Standardize features
        [features_SVM_Train, coeff_std] = standardizeData(features_SVM_Train);
        features_SVM_Test               = standardizeData(features_SVM_Test, coeff_std);

        % Get labels
        labels_SVM_Train = data_SVM_Train.Site;
        labels_SVM_Test  = data_SVM_Test.Site;

        % Prepare permutation labels
        tmp_labelsAll           = [labels_SVM_Train; labels_SVM_Test];
        numSamples              = length(tmp_labelsAll);
        randOrder               = randperm(numSamples, numSamples);
        tmp_labelsAll           = tmp_labelsAll(randOrder);
        labels_SVM_Perm_Train   = tmp_labelsAll(1:length(labels_SVM_Train));
        labels_SVM_Perm_Test    = tmp_labelsAll(length(labels_SVM_Train)+1:end);

        % Train classifier
        if multiClass
            mdl_SVM         = fitcecoc(features_SVM_Train, labels_SVM_Train,      'Coding', 'onevsone', 'Learners', 'svm');
            mdl_SVM_Perm    = fitcecoc(features_SVM_Train, labels_SVM_Perm_Train, 'Coding', 'onevsone', 'Learners', 'svm');
        else
            mdl_SVM         = fitcsvm(features_SVM_Train,  labels_SVM_Train,      'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
            mdl_SVM_Perm    = fitcsvm(features_SVM_Train,  labels_SVM_Perm_Train, 'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
        end

        % Record ground truth for posterity
        groundTruth_SVM_Train{repeats, fold}       = labels_SVM_Train;
        groundTruth_SVM_Test{repeats, fold}        = labels_SVM_Test;
        groundTruth_SVM_Perm_Train{repeats, fold}  = labels_SVM_Perm_Train;
        groundTruth_SVM_Perm_Test{repeats, fold}   = labels_SVM_Perm_Test;

        % Make predictions
        predictions_SVM_Train{repeats, fold}       = predict(mdl_SVM, features_SVM_Train);
        predictions_SVM_Test{repeats, fold}        = predict(mdl_SVM, features_SVM_Test);
        predictions_SVM_Perm_Train{repeats, fold}  = predict(mdl_SVM_Perm, features_SVM_Train);
        predictions_SVM_Perm_Test{repeats, fold}   = predict(mdl_SVM_Perm, features_SVM_Test);

        % Evaluate classifier performances
        accuracy_SVM_Train(repeats, fold)          = sum(strcmpi(predictions_SVM_Train{repeats, fold}, labels_SVM_Train))/length(labels_SVM_Train) * 100;
        accuracy_SVM_Test(repeats, fold)           = sum(strcmpi(predictions_SVM_Test{repeats, fold},  labels_SVM_Test))/length(labels_SVM_Test)   * 100;
        accuracy_SVM_Perm_Train(repeats, fold)     = sum(strcmpi(predictions_SVM_Perm_Train{repeats, fold},  labels_SVM_Perm_Train))/length(labels_SVM_Perm_Train) * 100;
        accuracy_SVM_Perm_Test(repeats, fold)      = sum(strcmpi(predictions_SVM_Perm_Test{repeats, fold},   labels_SVM_Perm_Test))/length(labels_SVM_Perm_Test)   * 100;
    end
    
    % Average over folds
    accuracy_SVM_Train_repeats(repeats,1)        = mean(accuracy_SVM_Train(repeats,:));
    accuracy_SVM_Test_repeats(repeats,1)         = mean(accuracy_SVM_Test(repeats,:));
    accuracy_SVM_Perm_Train_repeats(repeats,1)   = mean(accuracy_SVM_Perm_Train(repeats,:));
    accuracy_SVM_Perm_Test_repeats(repeats,1)    = mean(accuracy_SVM_Perm_Test(repeats,:));
end

for repeats = numRepeats+1:numPermsLC
    
   % Set seed
    rng(seeds(repeats), 'twister');
    
    % Separate SVM Test samples
    cv = cvpartition(dataTable.Site, 'KFold', 20, 'stratify', true);

    for fold = 1:20
        
        % Extract SVM test and remaining data
        data_SVM_Test  = dataTable(cv.test(fold),:);
        data_remaining = dataTable(cv.training(fold),:);

        % Split the remaining data into SVM train and NH learn
        cv2 = cvpartition(data_remaining.Site, 'Holdout', 70*numSites, 'stratify', true);
    
        % Extract SVM train and NH learn
        data_SVM_Train = data_remaining(cv2.test,:);
        data_rem_Learn = data_remaining(cv2.training,:);

        % Take currSampleSize from NH Learn
        if currSampleSize*numSites == height(data_rem_Learn)
            data_NH_Learn = data_rem_Learn;
        else
            % Take currSampleSize from NH Learn
            cv3           = cvpartition(data_rem_Learn.Site, 'HoldOut', currSampleSize*numSites, 'stratify', true);
            data_NH_Learn = data_rem_Learn(cv3.test,:);
        end
        
        % Extract features
        features_NH_Learn  = data_NH_Learn{:,  8:end};
        features_SVM_Train = data_SVM_Train{:, 8:end};
        features_SVM_Test  = data_SVM_Test{:,  8:end};

        % Extract site information
        site_NH_Learn     = data_NH_Learn.Site;
        site_SVM_Train    = data_SVM_Train.Site;
        site_SVM_Test     = data_SVM_Test.Site;
        
        % Find sites
        for sites = 1:length(allSites)
            loc_SVMTrain(:,sites) = ismember(data_SVM_Train.Site, allSites(sites)); 
            loc_SVMTest(:, sites) = ismember(data_SVM_Test.Site,  allSites(sites));
            loc_NHlearn(:, sites) = ismember(data_NH_Learn.Site,  allSites(sites));
        end 

        % Send to neuroHarmonize
        [features_NH_Learn, features_SVM_Train, features_SVM_Test] = doHarmonization_LC(features_NH_Learn, features_SVM_Train, features_SVM_Test,  ...
                                                                                        site_NH_Learn,     site_SVM_Train,     site_SVM_Test,      outDir); %#ok<ASGLU>
        % Standardize features
        [features_SVM_Train, coeff_std] = standardizeData(features_SVM_Train);
        features_SVM_Test               = standardizeData(features_SVM_Test, coeff_std);

        % Get labels
        labels_SVM_Train = data_SVM_Train.Site;
        labels_SVM_Test  = data_SVM_Test.Site;

        % Prepare permutation labels
        tmp_labelsAll           = [labels_SVM_Train; labels_SVM_Test];
        numSamples              = length(tmp_labelsAll);
        randOrder               = randperm(numSamples, numSamples);
        tmp_labelsAll           = tmp_labelsAll(randOrder);
        labels_SVM_Perm_Train   = tmp_labelsAll(1:length(labels_SVM_Train));
        labels_SVM_Perm_Test    = tmp_labelsAll(length(labels_SVM_Train)+1:end);

        % Train classifier
        if multiClass
            mdl_SVM_Perm   = fitcecoc(features_SVM_Train, labels_SVM_Perm_Train, 'Coding', 'onevsone', 'Learners', 'svm');
        else
            mdl_SVM_Perm   = fitcsvm(features_SVM_Train,  labels_SVM_Perm_Train, 'KernelFunction', 'linear', 'Standardize', false, 'BoxConstraint', 1);
        end

        % Record ground truth for posterity
        groundTruth_SVM_Train{repeats, fold}       = labels_SVM_Train;
        groundTruth_SVM_Test{repeats, fold}        = labels_SVM_Test;
        groundTruth_SVM_Perm_Train{repeats, fold}  = labels_SVM_Perm_Train;
        groundTruth_SVM_Perm_Test{repeats, fold}   = labels_SVM_Perm_Test;

        % Make predictions
        predictions_SVM_Train{repeats, fold}       = cell(length(labels_SVM_Train), 1);
        predictions_SVM_Test{repeats, fold}        = cell(length(labels_SVM_Test),  1);
        predictions_SVM_Perm_Train{repeats, fold}  = predict(mdl_SVM_Perm, features_SVM_Train);
        predictions_SVM_Perm_Test{repeats, fold}   = predict(mdl_SVM_Perm, features_SVM_Test);

        % Evaluate classifier performances
        accuracy_SVM_Train(repeats, fold)          = NaN;
        accuracy_SVM_Test(repeats, fold)           = NaN;
        accuracy_SVM_Perm_Train(repeats, fold)     = sum(strcmpi(predictions_SVM_Perm_Train{repeats, fold},  labels_SVM_Perm_Train))/length(labels_SVM_Perm_Train) * 100;
        accuracy_SVM_Perm_Test(repeats, fold)      = sum(strcmpi(predictions_SVM_Perm_Test{repeats, fold},   labels_SVM_Perm_Test))/length(labels_SVM_Perm_Test)   * 100;
    end
    
    % Average over folds
    accuracy_SVM_Train_repeats(repeats,1)         = mean(accuracy_SVM_Train(repeats,:));
    accuracy_SVM_Test_repeats(repeats,1)          = mean(accuracy_SVM_Test(repeats,:));
    accuracy_SVM_Perm_Train_repeats(repeats,1)    = mean(accuracy_SVM_Perm_Train(repeats,:));
    accuracy_SVM_Perm_Test_repeats(repeats,1)     = mean(accuracy_SVM_Perm_Test(repeats,:));    
end

% Calculate p value
pValues_test = zeros(numRepeats,1);
for repeats  = 1:numRepeats
    pValues_test(repeats,1) = (sum(accuracy_SVM_Perm_Test_repeats >= accuracy_SVM_Test_repeats(repeats,1)) + 1)/(numPermsLC + 1);
end
overall_pValue = mean(pValues_test);