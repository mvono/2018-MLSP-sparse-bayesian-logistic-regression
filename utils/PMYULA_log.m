
%-------------------------------------------------------------------------%
%                   SAMPLING THE SPLITTING VARIABLE z1                    %
%-------------------------------------------------------------------------%

function z1_new = PMYULA_log(z1,beta,u1,rho,y,XTrain,NTrain)

%-------------------------------------------------------------------------%
% This function samples the splitting variable z1 thanks to a proximal MCMC
% algorithm called P-MYULA (see Durmus et al., 2018).

    % INPUTS:
        % z1,beta,u1: current MCMC iterates.
        % rho: user-defined standard deviation of the variable of 
        %      interest beta.
        % y: responses taking values in {-1,1}.
        % XTrain: observation matrix associated to the training set.
        % NTrain: number of observations within the training set.
        
    % OUTPUT:
        % z1_new: new value for the splitting variable z1.
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% PRE-PROCESSING
lambda_MYULA = rho^2; % as prescribed in Durmus et al.
gamma_MYULA = (rho^2)/4; % as prescribed in Durmus et al.
z1_new = z1;

for j = 1:NTrain

% 1. SAMPLE THE ZERO-MEAN GAUSSIAN VARIABLE u.
u = randn(1,1);

% 2. UPDATE THE VALUE OF z1.
    % 2.1. Compute the gradient of f(z1) = (1 / (2 * rho^2)) ...
    % * ||z1 - (u1 + K*beta)||_2^2.
    gradz1 = (1 / rho^2) * (z1(j) - y(j) * XTrain(j,:) * beta - u1(j));
    
    % 2.2. Compute the proximal operator of g: prox(z1)^(lambda_MYULA)_g.
    w = rLambert(lambda_MYULA * exp(-z1(j)),exp(-z1(j)),6);
    if isnan(w) == 1 % preventing numerical problems with the
                     % exponentiation in the r-Lambert function
        w = lambda_MYULA * (1 - exp(lambda_MYULA + z1(j)) ...
            + (1 + lambda_MYULA) * exp(2 * (lambda_MYULA + z1(j))));
    end
    proxz1 = z1(j) + w;
    
    % 2.3. Compute the new value of z1: z1_new.
    z1_new(j) = (1 - gamma_MYULA / lambda_MYULA) * z1(j) ...
    - gamma_MYULA * gradz1 + (gamma_MYULA / lambda_MYULA) * proxz1 ...
    + sqrt(2 * gamma_MYULA) * u;

end
%-------------------------------------------------------------------------%

end