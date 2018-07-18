
%-------------------------------------------------------------------------%
%                PRE-PROCESSING STEP BEFORE LAUNCHING SPA                 %
%-------------------------------------------------------------------------%

function [labelsTest,D,NTrain,NTest,XTrain,XTest,y,Dy,Q,G] = ...
         preprocessing(cv,kfold,features,labels,lab1,i,rho)
     
%-------------------------------------------------------------------------%
% This function aims at preparing the data for the classification task made
% by SPA algorithm.

    % INPUTS:
        % cv: cvpartition class.
        % kfold: index of the current fold of the cross-validation.
        % features: observation matrix (colums: features; 
        %                               rows: observations).
        % labels: vector of labels associated to each observation.
        % lab1: 1st class of each binary classification problem 
        % (see initial_param_SPA.m file).
        % i: index associated to the current binary classification problem.
        % rho: hyperparameter associated to the variable splitting step.
        
    % OUTPUT:
        % labelsTest: vector of labels associated to each observation 
        % within the testing set.
        % D: dimensionality of the problem.
        % NTrain: number of observations within the training set.
        % NTest: number of observations within the testing set.
        % XTrain: observation matrix associated to the training set.
        % XTest: observation matrix associated to the testing set.
        % y: responses taking values in {-1, 1}.
        % Dy: NxN diagonal matrix with y_i as i-th diagonal element.
        % Q: pre-processed matrix used within the E-PO algorithm.
        % G: pre-processed matrix used within the E-PO algorithm.
%-------------------------------------------------------------------------%

% Define the training set
featuresTrain = features(cv.training(kfold),:);
labelsTrain = labels(cv.training(kfold),:);

% Define the test set
featuresTest = features(cv.test(kfold),:);
labelsTest = labels(cv.test(kfold),:);

% Define the dimensionality of the problem considered
D = size(features,2); % number of features
NTrain = size(featuresTrain,1); % number of training observations
NTest = size(featuresTest,1); % number of testing observations

% Define the observation matrix
XTrain = [ones(NTrain,1),featuresTrain]; % training set
XTest = [ones(NTest,1),featuresTest]; % test set

% Replace labels of the current class with 1 and other labels with
% -1.
idx = find(labelsTrain == lab1(i));
idx_other = find(labelsTrain ~= lab1(i));
y = labelsTrain;
y(idx) = 1;
y(idx_other) = -1;  

% Pre-processing
Dy = spdiags(y,0,NTrain,NTrain);
G = XTrain' * (Dy') * Dy * XTrain;
Q = (1 / rho^2) * (G + speye(D+1));
G = XTrain' * (Dy');
