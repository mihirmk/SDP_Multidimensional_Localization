function [X] = GradDesc(D, W)
n = size(D, 1);
e = ones(n, 1);

% Gradient Descent Parameters
t = 1e-3;
rho = 1e-4;
gamma = 0.85;
eta = 0.2;
numiters = 10000;
Q = 1;
% Initial Values
X = 20000 * (ones(n,n) + rand(n,n));
aF = (W .*(X - D))^2;
adF = 2 * transpose(X) * (W .* (X - D));

fF = norm(aF,'fro')^2;
fdF = norm(adF,'fro');

Cval = fF;
%-------------------------------------------------------------------------------
for i = 1 : numiters
    Xp = X;
    aFp = aF;
    adFp = adF;
    fFp = fF;
    fdFp = fdF;
    
    nls = 1;
    while 1
        X = Xp - t * adFp;
        X = projPSD(X);
        aF = (W .*(X - D));
        adF = 2 * transpose(X)* (W .* (X - D));
        fF = norm(aF,'fro')^2;
        fdF = norm(adF,'fro');
        c = rho*fdF^2;
        
        if (fF <= Cval - t*c) || (nls > 50)
            break;
        end
        t = eta*t;
        nls = nls + 1;
    end
    Qp = Q; 
    Q = gamma*Qp + 1; 
    Cval = (gamma*Qp*Cval + fF)/Q;    
end

end
function Qp = projPSD(Q) 
% This funtion return the symmetric positive semi-definite matrix Qp
% nearest to Q, based on eigen value decomposition.
    %Q = (Q + Q')/2; 
    [V,ee] = eig(Q); 
    d = max(diag(ee),0)';
    VD = bsxfun(@times,V, d);
    Qp = VD*V';
end