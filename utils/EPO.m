
%-------------------------------------------------------------------------%
%             E-PO ALGORITHM TO SAMPLE THE VARIABLE BETA                  %
%-------------------------------------------------------------------------%

function beta = EPO(z1,z2,u1,u2,rho,NTrain,D,G,Q)

%-------------------------------------------------------------------------%
% This function computes the E-PO algorithm as described in the paper of C.
% Gilavert et al., 2015. This algorithm deals with the exact resolution 
% case of the linear system Q*beta = eta and with a guaranteed convergence 
% to the target distribution.

    % INPUTS:
        % y: noisy observation (1D array).
        % H: direct operator in the linear inverse problem y = H*x + n.
        % sigma: user-defined standard deviation of the noise.
        % U,Z,delta: current MCMC iterates of the other variables.
        % rho: user-defined standard deviation of the variable of 
        %      interest x.
        % N,M: respectively, the dimension of X (2D-array) and y
        % (1D-array).
        % invQ: pre-computed covariance matrix involved in the posterior
        % distribution of the variable of interest x.
        
    % OUTPUT:
        % x: sample from the posterior distribution of x (2D-array).
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% 1. Sample eta from N(Q*mu,Q)
    % 1.1. Sample eta1 from N(z1-u1,rho^2*I)
    eta1 = z1 - u1 + rho*randn(NTrain,1);  

    % 1.2. Sample eta2 from N(z2-u2,rho^2*I)
    eta2 = z2 - u2 + rho*randn(D+1,1); 

    % 3. Set eta
    eta = (1 / rho^2) * (eta2 + G * eta1);
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% 2. Compute an exact solution beta_new of Q*beta = eta <=> beta = invQ*eta
beta = Q \ eta;
%-------------------------------------------------------------------------%

end