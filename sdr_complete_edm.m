function [EDM, X] = sdr_complete_edm(D, W, lambda)
n = size(D, 1);
x = -1/(n + sqrt(n));
y = -1/sqrt(n);
V = [y*ones(1, n-3); x*ones(n-1,n-3) + eye(n-1,n-3)];
e = ones(n, 1);

cvx_begin sdp
    variable G(n-3, n-3) symmetric;
    B = V*G*V';
    E = diag(B)*e' + e*diag(B)' - 2*B;
    maximize trace(G) ...
         - lambda * norm(W .* (E - D), 'fro');
    subject to
    G >= 0;
    %Ge = 0;
cvx_end

[U, S, V] = svd(B);
EDM = diag(B)*e' + e*diag(B)' - 2*B;
X = sqrt(S)*V';
end



