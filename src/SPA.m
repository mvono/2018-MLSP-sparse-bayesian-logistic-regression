
%-------------------------------------------------------------------------%
%                  SPLIT-AND-AUGMENTED GIBBS SAMPLER                      %
%-------------------------------------------------------------------------%

function [betaMC] = SPA(NTrain,D,rho,G,Q,alpha,y,XTrain,Dy,N_MC,tau)

%-------------------------------------------------------------------------%
% This function computes the SPA algorithm to solve the linear inverse 
% problem y = H*x + n associated to the image inpainting problem.

    % INPUTS:
        % NTrain: number of observations within the training set.
        % D: dimensionality of the problem.
        % rho: hyperparameter associated to the splitting step.
        % G: pre-processed matrix used within the E-PO algorithm.
        % Q: pre-processed matrix used within the E-PO algorithm.
        % alpha: hyperparameter associated to the data augmentation step.
        % y: responses taking values in {-1, 1}.
        % XTrain: observation matrix associated to the training set.
        % Dy: NxN diagonal matrix with y_i as i-th diagonal element.
        % N_MC: number of MCMC iterations.
        % tau: regularization parameter associated to the Laplacian prior.
        
    % OUTPUT:
        % betaMC: samples associated to the variable of interest beta.
%-------------------------------------------------------------------------%

tic;
disp(' ');
disp('BEGINNING OF THE SAMPLING');

%-------------------------------------------------------------------------
% Initialization
z1 = zeros(NTrain,1);
z2 = zeros(D+1,1);
u1 = zeros(NTrain,1);
u2 = zeros(D+1,1);
betaMC = [];
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
% Gibbs sampling
h = waitbar(0,'Sampling in progress...');
for t = 1:N_MC
    
    % 1. Sample the vector of weights beta from p(beta|z1,z2,u1,u2) using 
    % Exact Perturbation-Optimization (E-PO) method.
    beta = EPO(z1,z2,u1,u2,rho,NTrain,D,G,Q);
    
    % 2. Sample z2 from p(z2|beta,u2) using P-MYULA 
    % (see Durmus et al., 2018).
    z2 = PMYULA_l1Norm(z2,beta,u2,rho,D,tau);
    
    % 3. Sample u2 from p(u2|beta,z2).
    u2 = (alpha^2 /(alpha^2 + rho^2)) * (z2 - beta) ...
          + randn(D+1,1) * sqrt(alpha^2 + rho^2)/(alpha*rho);
      
    % 4. Sample z1 from p(z1|beta,u1) using P-MYULA 
    % (see Durmus et al., 2018). [This step can be processed in parallel]
    z1 = PMYULA_log(z1,beta,u1,rho,y,XTrain,NTrain);
    
    % 5. Sample u1 from p(u1|beta,z1).
    u1 = (alpha^2 /(alpha^2 + rho^2)) * (z1 - Dy * XTrain * beta) ...
          + randn(NTrain,1) * sqrt(alpha^2 + rho^2)/(alpha*rho);
    
    % 6. Store the iterates of the variable of interest beta.
    betaMC = [betaMC beta];
    
    % 7. Show iteration counter
    waitbar(t/N_MC,h);

end
%-------------------------------------------------------------------------

t_1 = toc;
close(h);
disp('END OF THE GIBBS SAMPLING');
disp(['Execution time of the Gibbs sampling: ' num2str(t_1) ' sec']);

end