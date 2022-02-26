function [G] = proj_grad(D, W)
n = size(D, 1);
x = -1/(n + sqrt(n));
y = -1/sqrt(n);
V = [y*ones(1, n); x*ones(n) + eye(n)];
e = ones(n, 1);

% [U, S, V] = svd(B);
% EDM = diag(X)*e' + e*diag(X)' - 2*X;
% X = sqrt(S)*V';

%-------------------------------------------------------------------------------
%% Parameterization
xtol = 1e-5;
Aisdiag = 0;
ftol = 1e-12;
% parameters for control the linear approximation in line search
rho = 1e-4;
% factor for decreasing the step size in the backtracking line search
eta = 0.2;
% parameters for updating C by HongChao, Zhang
tau = 1e-3;
mxitr = 1000;    

%-------------------------------------------------------------------------------
%% Initial function value and gradient
% Initial Values
G = 1000 * ones(n,n);
E = diag(G)*e' + e*diag(G)' - 2*G;
F = (W .*(E - D));
dF = 2 * transpose(E)* (W .* (E - D));

aF = norm(F,'fro')^2;
adF = norm(dF,'fro');

%% main iteration
tstart = tic;
for itr = 1 : mxitr
    GP = G;    
    FP = F;  
    dFP = dF;   
    
    % Backtracking
    nls = 1; 
    deriv = rho*adF^2; 
    while 1     
        % Update scheme:
        G = GP - tau * dFP; 

        % Calculate F
        E = diag(G)*e' + e*diag(G)' - 2*G;
        F = (W .* (E - D));
        aF = norm(F,'fro')^2;
        if nls >= 5
            break;
        end
        tau = eta*tau;          
        nls = nls + 1;
    end
    
    % Calculate the new gradient:
    dF = 2 * transpose(E) * (W .* (E - D));
    adF = norm(dF,'fro');        
    
    % Calculate the Barzilai-Borwein step-size:
    tau = max(min(tau, 1e20), 1e-20);

%     % Stopping Rules:  
%     if( NrmS < xtol )  
%         if itr <= 2
%             ftol = 0.1*ftol;
%             xtol = 0.1*xtol;
%         else
%             msg = 'converge';
%             break;
%         end
%     end
%     
%     Qp = Q; 
%     Q = gamma*Qp + 1; 
%     Cval = (gamma*Qp*Cval + F)/Q;
end
time = toc(tstart);


end

