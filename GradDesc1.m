function [X] = GradDesc1(D, W)
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
X = 10000 * rand(n,n);
aF = (W .*(X - D))^2;
adF = 2 * (W .* (X - D)) * transpose(X);

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
        aF = (W .*(X - D));
        adF = 2 * (W .* (X - D)) * transpose(X);
        fF = norm(aF,'fro')^2;
        fdF = norm(adF,'fro');
        c = rho*fdF^2;

        gradDt = coords - Zt2;
        gradDtn = sqrt((meshgrid(gradDt(:,1))-meshgrid(gradDt(:,1)).').^2 + (meshgrid(gradDt(:,2))-meshgrid(gradDt(:,2)).').^2);
        X = X0 - t*gradDtn;
        
        if (fF <= Cval - t*c) || (nls > 50)
            break;
        end
        t = eta*t;
        nls = nls + 1;
    end
end

end