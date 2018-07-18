
%-------------------------------------------------------------------------%
%                    SPLIT-AND-AUGMENTED GIBBS SAMPLER (SPA)              %
%                 APPLIED TO HANDWRITTEN DIGITS CLASSIFICATION            %
%                        ON MNIST DATABASE (ONE-VS-ALL)                          %
%-------------------------------------------------------------------------%
% File: SPA_MNIST.m
% Author: M. VONO
% Created on: 18/07/2018
% Last modified : 18/07/2018
clearvars;
close all;
addpath('../utils/'); % to use P-MYULA within SPA and other functions
addpath('../src/'); % to launch SPA
%-------------------------------------------------------------------------%
% REF.                                                                    %
% M. VONO et al.,                                                         %
% "Sparse Bayesian binary logistic regression                             %
% using the split-and-augmented Gibbs sampler", IEEE International        %
% Workshop on Machine Learning for Signal Processing (MLSP),              %
% Aalborg, Denmark, 2018.                                                 %
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% Load workspace variables (go to ../utils/initial_param_SPA.m to 
% modify them), prepare the classification task (pre-processing, 
% train/test sets definition, etc.) and launch SPA algorithm.                                 
load('../utils/initial_param_SPA.mat'); 

% Load dataset
features = loadMNISTImages('../utils/dataset/MNIST/train_images')';
labels = loadMNISTLabels('../utils/dataset/MNIST/train_labels');
Nclass = length(unique(labels)); % number of differents classes

% For each k-fold cross-validation procedure
for nrep = 1:Nrep
    rng(nrep);
    disp(['Repetition n° ' num2str(nrep) '.'])
    
    % Partition dataset into k folds
    cv = cvpartition(labels,'KFold',k);

    % For each fold
    for kfold = 1:k
            
        % Define the training set
        featuresTrain = features(cv.training(kfold),:);
        labelsTrain = labels(cv.training(kfold),:);

        % Define the test set
        featuresTest = features(cv.test(kfold),:);
        labelsTest = labels(cv.test(kfold),:);
            
        % For each class, build a binary classifier
        for i = 0:Nclass-1 
            
            % Preprocessing step
            [D,NTrain,NTest,XTrain,XTest,y,Dy,Q,G] = ...
    preprocessing_onevsall(labelsTrain,featuresTrain,featuresTest,i,rho);
            
            % Launch SPA algorithm
            betaMC = SPA(NTrain,D,rho,G,Q,alpha,y,XTrain,Dy,N_MC,tau);
            
            % For each sample beta, calculate the probability to belong to the
            % first class
            logproba = XTest * betaMC - log(1 + exp(XTest * betaMC));
            proba = exp(logproba);

            % Average probability
            probaMMSE(i+1,:) = mean(proba(:,N_bi+1:end),2);
            betaMMSE(i+1,:) = mean(betaMC(:,N_bi+1:end),2);

            % 90% credibility intervals
            quant_5 = zeros(NTest,1);
            quant_95 = zeros(NTest,1);
            for t = 1:NTest
                    arr = proba(t,N_bi+1:end);
                    quant_5(t) = quantile(arr,0.05);
                    quant_95(t) = quantile(arr,0.95);
            end
            probaUP(i+1,:) = quant_95;
            probaDOWN(i+1,:) = quant_5;
            
            disp(['Binary classifier n° ' num2str(i+1) ' trained.']);
        
        end

        % Prediction on test set
        [~,loc] = max(probaMMSE);
        prev = (loc - 1)';
        err  = abs(prev - labelsTest);
        idx_err = find(err > 0);
        errRate(nrep,kfold) = length(idx_err)/NTest;

        % Give a 90% confidence on the classification made by the 
        % classifier
        probaDOWN = probaDOWN';
        I = (1 : size(probaDOWN, 1)) .';
        J = reshape(loc', [], 1);
        id = sub2ind(size(probaDOWN), I, J);
        C = probaDOWN(id);
        probaUP = probaUP';
        idxUP = zeros(NTest,Nclass);
        countUncertainty = 0;
        idxUncertainty = [];
        for kk = 1:NTest
            loc = find(probaUP(kk,:) >= C(kk));
            idxUP(kk,loc) = 1;
            if length(loc) > 1
                countUncertainty = countUncertainty + 1;
                idxUncertainty = [idxUncertainty;kk];
            end
        end     
        confidence90(nrep,kfold) = countUncertainty/NTest;
        disp(['Classification error rate: ' ...
          num2str(100*errRate(nrep,kfold)) ' % with ' ...
          num2str(100*confidence90(nrep,kfold)) ...
          ' % possible missclassified digits under 90% credibility intervals']);

        % Error rate without the possible missclassifications
        err(idxUncertainty) = [];
        idx_err = find(err > 0);
        errRateWithout(nrep,kfold) = length(idx_err)/NTest;
        disp(['Error rate without possible missclassification: ' ...
              num2str(100*errRateWithout(nrep,kfold)) ' %.']);

        disp(['Iteration ' num2str(nrep) ' for fold ' ...
               num2str(kfold) ' finished.']);
           
       clear probaMMSE probaUP probaDOWN betaMMSE;

    end

    % Displaying the average classification error over the k folds
    errRate_avg = mean(errRate(nrep,:),2);
    disp(['Average classification error rate over the ' num2str(k) ...
      ' folds: ' num2str(100*errRate_avg) ' %.']);
end
%-------------------------------------------------------------------------%