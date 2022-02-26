function demo_PSDP(n, m, problem, Aisdiag)
%-------------------------------------------------------------------------
% A demo of solving (PSDP)
%   min 0.5||XA - B||, s.t., X is a n-by-n symmetric positive semi-deifinite matrix. 
%
%  where A and B are a given n-by-m matrices with n>=m.
%  INPUT: 
%         - m and n: are de dimension of A and B matrices. (n>=m)
%         - problem: is have two options {1,2}. If problem == 1 then A is
%           generated randomly with a well condition number, otherwise, if 
%           problem == 2 then A es generated randomly with a ill condition
%           number.
%         - Aisdiag: is a parameter with two options {1,0}. If Aisdiag == 1
%           then the matrix A is generated randomly as a diagonal matrix, 
%           otherwise, if Aisdiag == 0, A is generated randomly as a 
%           dense matrix.
%
%  This demo generate a PSDP randomly and solve it.
%-------------------------------------------------------------------------
% 
% Example1:
% demo_PSDP()
% 
% Example2:
% demo_PSDP(1000,1000,1,0)
%
% Reference:
% ----------
%  Harry F. Oviedo Leon
%  "Un M?todo de Gradiente Projectado Espectral para el Problema Procrustes 
%  Semidefinido Positivo". (ResearchGate)
%
%  In english: "Spectral Projected Gradient Method for the positive 
%               semidefinite Procrustes problem".
%
% Authors:
% ----------
% Harry F. Oviedo Leon <harry.oviedo@cimat.mx>
%
% Date: 01-Oct-2017
%-------------------------------------------------------------------------
clc
if nargin < 1
    n = 500; m = 500; problem = 1; Aisdiag = 0;   
end
%-------------------------------------------------------------------------
% Generating the problem 
%-------------------------------------------------------------------------
% Generating A:
S = zeros(n,m);    
if( problem == 1 )
    for i = 1:m
        S(i,i) = 2*rand + 10;   
    end
elseif( problem == 2 )
    for i = 1:m
        S(i,i) = i + 2*rand; 
    end 
end
if Aisdiag == 1
    A = S;
else 
    M1    = randn(n);        M2 = randn(m);
    [Q,~] = qr(M1);       [P,~] = qr(M2);
    A = Q*S*P';
end
% Generating the Global Solution:
M1    = rand(n,n);
[Q,~] = qr(M1);
d     = rand(n,1);      d(1) = 0;   d(2) = 0;
X_opt = Q*diag(d)*Q';
% Generating the initial point:
M1    = randn(n,n);      
X0    = M1'*M1;
% Generating B:
B = X_opt*A;
%-------------------------------------------------------------------------
% Solving the problem with our solver 
%-------------------------------------------------------------------------
opts.record = 0;
opts.mxitr  = 5000;
opts.xtol = 1e-5;
opts.ftol = 1e-12;
opts.tau  = 1e-3;
opts.Proj = 1;
opts.gamma = 0.85;
if Aisdiag == 1
    opts.Aisdiag = 1;
else
    opts.Aisdiag = 0;
end
%--- Our solver ---
[X, F_eval, out] = OptPSDPwithBB(A, B, X0, opts);
%-------------------------------------------------------------------------
% Printing the results
%-------------------------------------------------------------------------
[~,e] = eig(X);     
fprintf('Results\n') 
fprintf('---------------------------------------------------\n')
fprintf('   Obj. function = %7.6e\n',  out.fval);
fprintf('   Min. eigenvalue of X = %3.4f \n', min(diag(e)));
fprintf('   Distance ||X-X^{T}||_F = %7.6e \n',norm(X-X','fro'));
fprintf('   Iteration number = %d\n',  out.itr); 
fprintf('   Cpu time (secs) = %3.4f  \n',  out.time);
fprintf('   Absolute Error in X: ||X_{k+1}-X_{k}||_F) = %d\n',  out.Errabsx); 
fprintf('   Global Error = %3.2e\n',norm(X - X_opt,'fro'));
%-------------------------------------------------------------------------
figure, plot(F_eval,'linewidth',1.5)
xlabel('Iterations (k) ')
ylabel('Obj. function evaluations at (X_k)')
end
