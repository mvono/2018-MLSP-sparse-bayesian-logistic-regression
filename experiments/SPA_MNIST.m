
%-------------------------------------------------------------------------%
%                    SPLIT-AND-AUGMENTED GIBBS SAMPLER (SPA)              %
%                 APPLIED TO HANDWRITTEN DIGITS CLASSIFICATION            %
%                              ON MNIST DATABASE                          %
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

% For each k-fold cross-validation procedure
for nrep = 1:Nrep
    rng(nrep);
    disp(['Repetition n° ' num2str(nrep) '.'])

    % For each MNIST binary classification problem
    for i = 1:Nexample

        % Load dataset
        load(['../utils/dataset/MNIST/ini_' num2str(lab1(i)) ...
            num2str(lab2(i)) '.mat']);

        % Partition dataset into k folds
        cv = cvpartition(labels,'KFold',k);
        
        % For each fold
        for kfold = 1:k
            
            % Preprocessing step
            [labelsTest,D,NTrain,NTest,XTrain,XTest,y,Dy,Q,G] = ...
         preprocessing(cv,kfold,features,labels,lab1,i,rho);
            
            % Launch SPA algorithm
            betaMC = SPA(NTrain,D,rho,G,Q,alpha,y,XTrain,Dy,N_MC,tau);
            
            % For each sample beta, calculate the probability to belong to 
            % the first class
            logproba = XTest * betaMC - log(1 + exp(XTest * betaMC));
            proba = exp(logproba);

            % Average probability
            probaMMSE = mean(proba(:,N_bi:N_MC),2);

            % 90% credibility intervals
            CI_90 = zeros(NTest,1);
            for t = 1:NTest
                    arr = proba(t,N_bi:end);
                    quant_5 = quantile(arr,0.5);
                    quant_95 = quantile(arr,0.95);
                    CI_90(t) = abs(quant_95 - quant_5);
            end
            probaUP = probaMMSE + CI_90;
            probaDOWN = probaMMSE - CI_90; 

            % Compute the prediction on the test set with the average
            % probability
            idx = find(probaMMSE >= 0.5);
            idx_other = find(probaMMSE < 0.5);
            prev = ones(NTest,1);
            prev(idx) = lab1(i); 
            prev(idx_other) = lab2(i); 
            err  = abs(prev - labelsTest);
            idx_err = find(err > 0);
            errRate(i,nrep,kfold) = length(idx_err)/NTest;

            % Give a 90% confidence on the classification made by the 
            % classifier
            idxUP = find(probaUP >= 0.5 & probaMMSE < 0.5);
            idxDOWN = find(probaDOWN < 0.5 & probaMMSE >= 0.5);        
            confidence90(i,nrep,kfold) = length([idxDOWN;idxUP])/NTest;
            disp(['Classification error rate: ' ...
              num2str(100*errRate(i,nrep,kfold)) ' % with ' ...
              num2str(100*confidence90(i,nrep,kfold)) ...
              ' % possible missclassified digits under 90% credibility' ...
              ' intervals']);

            % Error rate without the possible missclassifications
            idx = [idxDOWN;idxUP];
            err(idx) = [];
            idx_err = find(err > 0);
            errRateWithout(i,nrep,kfold) = length(idx_err)/NTest;
            disp(['Error rate without possible missclassification: ' ...
                  num2str(100*errRateWithout(i,nrep,kfold)) ' %.']); 

            disp(['Iteration ' num2str(i) ' for fold ' num2str(kfold) ...
                  ' finished.']);
        end

        % Displaying the average classification error over the k folds
        errRate_avg = mean(errRate(i,nrep,:),3);
        disp(['Average classification error rate over the ' num2str(k) ...
          ' folds: ' num2str(100*errRate_avg) ' %.']);
    end
    
end
%-------------------------------------------------------------------------%