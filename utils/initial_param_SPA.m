
%-------------------------------------------------------------------------%
%                        INITIAL VARIABLES AND PARAMETERS                 %
%-------------------------------------------------------------------------%

clearvars;

% parameters of the binary classification
Nexample = 3; % number of binary classification problems
lab1 = [1 4 3]; % 1st label for each binary classification problem
lab2 = [7 6 5]; % 2nd label for each binary classification problem

% number of repetitions of the k-fold cross-validation procedure
Nrep = 3;

% define the k-fold cross-validation strategy
k = 5;

% parameters of the Gibbs sampler SPA
N_MC = 20; % total number of MCMC iterations
N_bi = 2; % number of burn-in iterations
rho = 3; % penalty hyperparameter
alpha = 1; % hyperparameter associated to the prior on the auxiliary 
           % variable u
tau = 1; % regularization parameter

% save the initial parameters
save('initial_param_SPA.mat');

% check that the file is saved
fprintf('Initial parameters saved! \n');
