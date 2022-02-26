function [X, E, out] = OptPSDPwithBB(A, B, X, opts)
%-------------------------------------------------------------------------
% Spectral Projected Gradient Method.  
%
% Given n-by-m matrices A and B, this code solves numerically the 
% following constrained optimization problem:  
%
%    F(X) = 0.5||AX - B||_F^{2}   s.t.  X\in S_{+}(n)          (1)
%
% where S_{+}(n) is the set form by all symmetric and positive semidefinite
% matrices n-by-n with real entries.
%
% This code is an implementation of a non-monotone projected gradiente, 
% whose update formula is:
%
%    X_{k+1} = \pi( X_k - \tau grafF(X_k) ),
%
% where \pi() is an operator proyection onto S_{+}(n). 
% ------------------------------------------------------------------------
%
% INPUT:
%           X --- n-by-n matrix such in S_{+}(n), (this is the initial point)
%           A --- n-by-m  matrix. 
%           B --- n-by-m  matrix.
%        opts --- option structure with fields:
%                 record = 0, No print out.
%                 mxitr       Max number of iterations.
%                 xtol        Stop control for ||X_k - X_{k-1}||_F.
%                 ftol        Stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                 Proj = 1,   If Proj==1 then the projection operator use 
%                             Cholesky in order to reduce the computacional 
%                             cost, otherwise, if Proj == 0 the procedure
%                             use.
%                             spectral decomposition in all iterations.
%                 Aisdiag ,   Aisdiag == 1 if A is a diagonal matrix, and
%                             Aisdiag == 0 if A is a dense matrix.
%                             Aisdiag = 0 for default.
%                 gamma,      Is a parameter of the non-monotone technique
%                             proposed by Zhang-Hager.
%                 rho         Parameter for control the linear approximation 
%                             in line search.
%                 eta         Factor for decreasing the step size in the 
%                             backtracking line search.
%                 tau         The initial step size.
% 
% OUTPUT:
%           X --- approximate solution
%         Out --- output information
%           E --- Objective function evaluations at X_k.
% ------------------------------------------------------------------------
% For example, consider the following problem: 
% 
%   Creating the example:
% n = 50;      m = 100;
% A = randn(n,m);      
% M = rand(n,n);       [Q,~] = qr(M);       
% d = rand(n);          d(1) = 0;
% X_opt = Q'*diag(d)*Q;     % Note that: X_opt belongs to S_{+}(n).
% B = X_opt*A;              % Observe that: F(X_opt) = 0.
% M = rand(n,n);        
% X0 = M'*M;
%
%   Setting the optionals parameters:
% opts.record = 0; 
% opts.mxitr  = 1000;
% opts.xtol = 1e-5;
% opts.ftol = 1e-12;
% opts.Proj = 0;
%
%   Solving the problem:
%[X, out]= OptPSDPwithBB(A, B, X0, opts);
%fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(X-X_opt): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, out.time, norm(X - X_opt, 'fro') );
% 
% ------------------------------------------------------------------------
%
% Reference: 
%  Harry F. Oviedo 
%  "Un M?todo de Gradiente Projectado Espectral para el Problema Procrustes 
%  Semidefinido Positivo". (ResearchGate)
%
%  In english: "Spectral Projected Gradient Method for the positive 
%               semidefinite Procrustes problem".
%
% Author: Harry F. Oviedo
%   Version 1.0 .... 2017/10
%-------------------------------------------------------------------------
%% Size information
if isempty(A)
    error('input A is an empty matrix');
else
    [n,m] = size(A);
end
if isempty(B)
    error('input B is an empty matrix');
end
if isempty(X)
    error('input X is an empty matrix');
end
if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-5;
    end
else
    opts.xtol = 1e-5;
end
if isfield(opts, 'Aisdiag')
    if opts.Aisdiag ~= 0 && opts.Aisdiag ~= 1
        opts.Aisdiag = 0;
    end
else
    opts.Aisdiag = 0;
end
if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-12;
    end
else
    opts.ftol = 1e-12;
end
% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
   if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
   end
else
    opts.rho = 1e-4;
end
% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
   if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
   end
else
    opts.eta = 0.2;
end
% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
   if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
   end
else
    opts.gamma = 0.85;
end
if isfield(opts, 'tau')
   if opts.tau < 0 || opts.tau > 1e3
        opts.tau = 1e-3;
   end
else
    opts.tau = 1e-3;
end
if isfield(opts, 'mxitr')
    if opts.mxitr < 0 || opts.mxitr > 2^20
        opts.mxitr = 1000;     
    end
else
    opts.mxitr = 1000;    
end
if ~isfield(opts, 'record')
    opts.record = 0;
end
if isfield(opts, 'Proj')
    if opts.Proj ~= 0 && opts.xtol ~= 1
        opts.Proj = 1;
    end
else
    opts.Proj = 1;
end


%-------------------------------------------------------------------------------
% copy parameters
xtol    = opts.xtol;
ftol    = opts.ftol;
rho     = opts.rho;
eta     = opts.eta;
gamma   = opts.gamma;
Proj    = opts.Proj;
record  = opts.record;
Aisdiag = opts.Aisdiag;   
%% Initial function value and gradient
% prepare for iterations
if Aisdiag == 1
    a    = diag(A)';
    XA   = bsxfun(@times,X(:,1:m), a);
    hold = XA - B; 
    Grad = zeros(n,n);
    G = bsxfun(@times,hold, a);
    Grad(:,1:m) = G;
else
    hold = X*A - B;                 At = A';
    Grad = hold*At;
end
F = 0.5*norm(hold,'fro')^2;     
nrmG  = norm(Grad, 'fro');
out.nfe = 1;
Q = 1; Cval = F;  tau = opts.tau;
%% Print iteration header if debug == 1
if (opts.record == 1)
    fid = 1;
    fprintf(fid, '----------- Spectral Gradient Method with Line search ----------- \n');
    fprintf(fid, '%4s %8s %8s\n', 'Iter', 'tau', 'F(X)');
end
E = zeros(opts.mxitr+1,1);
E(1) = F;
%% main iteration
tstart = tic;
for itr = 1 : opts.mxitr
    XP = X;     FP = F;   GradP = Grad;   
    
    % Backtracking
    nls = 1; deriv = rho*nrmG^2; 
    while 1     
        % Update scheme:
        X = XP - tau*GradP; 
        %X = projPSD(X, Proj);
          
        % Calculate F,
        if Aisdiag == 1
            XA   = bsxfun(@times,X(:,1:m), a);
            hold = XA - B;
        else
            hold = X*A - B;
        end
                         
        F = 0.5*norm(hold,'fro')^2;     
        out.nfe = out.nfe + 1;
        
        if F <= Cval - tau*deriv || nls >= 5
            break;
        end
        tau = eta*tau;          nls = nls + 1;
    end
    
    % Calculate the new gradient:
    if Aisdiag == 1
        Grad = zeros(n,n);
        G = bsxfun(@times,hold, a);
        Grad(:,1:m) = G;
    else
        Grad = hold*At;
    end
    nrmG  = norm(Grad, 'fro');          
    E(itr+1) = F;
    % Calculate the Barzilai-Borwein step-size:
    S = X - XP;                          NrmS = norm(S,'fro');
    XDiff = NrmS/norm(X,'fro');          FDiff = abs(FP - F)/(abs(FP) + 1);
   
    Y = Grad - GradP;       SY = abs(sum(sum(S.*Y)));
    if mod(itr,2)==0; tau = NrmS^2/SY;
    else tau  = SY/sum(sum(Y.*Y)); end
    tau = max(min(tau, 1e20), 1e-20);
    % Printing the information of the iteration i:
    if (record >= 1)
        fprintf('%4d  %3.2e  %4.3e %3.2e  %3.2e  %2d  %3.2e\n', ...
            itr, tau, F, XDiff, FDiff, nls, NrmS);
    end
    
    % Stopping Rules:  
    if( ( NrmS < xtol ) || (XDiff < 100*ftol && FDiff < ftol) )  
        if itr <= 2
            ftol = 0.1*ftol;
            xtol = 0.1*xtol;
        else
            out.msg = 'converge';
            break;
        end
    end
    
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
end
time = toc(tstart);
E = E(1:itr+1);
if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end
%X = projPSD(X, Proj);
out.itr = itr;
out.time = time;
out.Errabsx = norm(X - XP,'fro');
out.fval = F;
end
function Qp = projPSD(Q, Proj) 
% This funtion return the symmetric positive semi-definite matrix Qp
% nearest to Q. If Proj == 1 then the procedure use Cholesky factorization
% in order to save the calculation of spectral decomposition if this is
% possible; otherwise (Proj == 0), then this function always calculate the spectra
% decomposition. 
    Q = (Q + Q')/2; 
    if max(max(isnan(Q))) == 1 || max(max(isinf(Q))) == 1
        error('Input matrix has infinite or NaN entries');
    end
    
    if Proj == 1
        try 
            L = chol(Q);    Qp = Q;    
        catch 
            [V,e] = eig(Q); 
            d = max(diag(e),0)';
            VD = bsxfun(@times,V, d);
            Qp = VD*V'; 
        end
    else
        [V,e] = eig(Q); 
        d = max(diag(e),0)';
        VD = bsxfun(@times,V, d);
        Qp = VD*V';
    end     
end