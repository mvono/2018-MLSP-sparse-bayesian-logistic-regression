
%-------------------------------------------------------------------------%
%                PRE-PROCESSING STEP BEFORE LAUNCHING SPA                 %
%-------------------------------------------------------------------------%

function [D,NTrain,NTest,XTrain,XTest,y,Dy,Q,G] = ...
    preprocessing_onevsall(labelsTrain,featuresTrain,featuresTest,i,rho)
     
%-------------------------------------------------------------------------%
% This function aims at preparing the data for the classification task made
% by SPA algorithm (one-vs-all approach).

    % INPUTS:
        % labelsTrain: vector of labels associated to each observation 
        % within the training set.
        % featuresTrain: observation matrix associated to the training set.
        % featuresTest: observation matrix associated to the testing set.
        % i: index associated to the current binary classification problem.
        % rho: hyperparameter associated to the variable splitting step.
        
    % OUTPUT:
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

% Find the indices of observations associated to class i
idx = find(labelsTrain == i);
idx_other = find(labelsTrain ~= i);

% Define the responses vector
y = labelsTrain;
feat = featuresTrain;
y(idx) = 1;
y(idx_other) = -1;

% Define the dimensionality of the problem considered
D = size(feat,2); % number of features
NTrain = size(feat,1); % number of training observations
NTest = size(featuresTest,1); % number of testing observations

% Define the observation matrix
XTrain = [ones(NTrain,1),feat]; % training set
XTest = [ones(NTest,1),featuresTest]; % test set

% Pre-processing
Dy = spdiags(y,0,NTrain,NTrain);
G = XTrain' * (Dy') * Dy * XTrain;
Q = (1 / rho^2) * (G + speye(D+1));
G = XTrain' * (Dy');
