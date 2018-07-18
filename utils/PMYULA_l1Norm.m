
%-------------------------------------------------------------------------%
%                   SAMPLING THE SPLITTING VARIABLE Z2                    %
%-------------------------------------------------------------------------%

function z2_new = PMYULA_l1Norm(z2,beta,u2,rho,D,tau)

%-------------------------------------------------------------------------%
% This function samples the splitting variable z2 thanks to a proximal MCMC
% algorithm called P-MYULA (see Durmus et al., 2018).

    % INPUTS:
        % z2,beta,u2: current MCMC iterates.
        % rho: user-defined standard deviation of the variable of 
        %      interest beta.
        % D: the dimension of beta.
        % tau: regularization parameter associated to the Laplacian prior.
        
    % OUTPUT:
        % z2_new: new value for the splitting variable z2.
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% PRE-PROCESSING
lambda_MYULA = rho^2; % as prescribed in Durmus et al.
gamma_MYULA = (rho^2)/4; % as prescribed in Durmus et al.

% 1. SAMPLE THE ZERO-MEAN GAUSSIAN VARIABLE u.
u = randn(D+1,1);

% 2. UPDATE THE VALUE OF z2.
    % 2.1. Compute the gradient of f(z2) = (1 / (2 * rho^2)) ...
    % * ||z2 - (u2 + beta)||_2^2.
    gradf = (1 / rho^2) * (z2 - u2 - beta);
    
    % 2.2. Compute the proximal operator of g: prox(z2)^(lambda_MYULA)_g.
    proxg = z2;
    proxg(2:end) = sign(proxg(2:end)) .* max(abs(proxg(2:end)) ...
                    - tau * lambda_MYULA,0);
    
    % 2.3. Compute the new value of z2: z2_new.
    z2_new = (1 - gamma_MYULA / lambda_MYULA) * z2 - gamma_MYULA * gradf...
        + (gamma_MYULA / lambda_MYULA) * proxg + sqrt(2 * gamma_MYULA) * u;

%-------------------------------------------------------------------------%

end