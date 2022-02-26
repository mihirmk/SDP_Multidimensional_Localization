clear all
close all
load mds_train


%% Estimation based on distance and time matrices with classic MDS

% Estimation based on distance matrix
[Yd,eigvalsd] = cmdscale(distance);
[Dd,Zd] = procrustes(coords,Yd);

% Estimation based on time matrix
[Yt,eigvalst] = cmdscale(time_matrix,2);
[Dt,Zt] = procrustes(coords,Yt);


% Graphs of real positions and estimates
figure(1)
hold on
%Real coordinates
plot(coords(:,1),coords(:,2),'X',...
    'MarkerSize',10);
%Estimated coordinates with distance matrix
plot(Zd(:,1),Zd(:,2),'.',...
    'MarkerSize',10);
%Estimated coordinates with times matrix
plot(Zt(:,1),Zt(:,2),'.',...
    'MarkerSize',10);
%Pimping graphic
text(coords(:,1)+0.05,coords(:,2),station_index)
legend( 'Known coordinates' , 'Distance matrix estimation' , 'Time matrix estimation' , 'Location', 'NorthWest')
title('Classic MDS Location Estimation')

%% EDM (Algorithm 5 in the reference article) %%
Dt = time_matrix;

n = size(Dt,1);
x = -1/(n+sqrt(n));
ye = -1/sqrt(n);
V = [ye*ones(1,n-1);x*ones(n-1)+eye(n-1)];
e = ones(n,1);
W = ones(n,n);
lambda = 1; %REVISAR COMO ELEGIR LAMBDA

cvx_begin sdp
    variable H(n-1,n-1) symmetric;
    B = V*H*V';
    E = diag(B)*e' + e*diag(B)'-2*B;
    maximize trace(H)-lambda*(norm(W.*(E-Dt),'fro'));
    subject to
        H >= 0;
cvx_end

%Calculate the estimated coordinates
EDM = diag(B)*e'+e*diag(B)'-2*B;
[Yt1,eigvalst] = cmdscale(EDM,2);
[Dt,Zt1] = procrustes(coords,Yt1);

%Graphs of real positions and estimates
figure(2)
hold on
%Real coordinates
plot(coords(:,1),coords(:,2),'X',...
    'MarkerSize',10);
%Estimated coordinates with SDR matrix
plot(Zt1(:,1),Zt1(:,2),'.',...
    'MarkerSize',10);
%Estimated coordinates with time matrix
plot(Zt(:,1),Zt(:,2),'.',...
    'MarkerSize',10);
%Pimping graphic
text(coords(:,1)+0.05,coords(:,2),station_index)
legend( 'Known coordinates' , 'Convex estimation' , 'Time matrix estimation' , 'Location', 'NorthWest')
title('MDS Location Estimation')

%% Gradient Descent
%Initialization
figure(3)
hold on
%X_0
X0 = time_matrix;
%Step size
t = 5;
%Stop condition
stop = cvx_slvtol;
%a = stop + 1;
%Number of iteration
it = 0;

X = zeros(size(X0));
Zt2 = Zt;

%Iterations
for a = 1:2
    %CPU time
    time = cputime;
    
    %Gradient descent
    %gradDt = 1-; %derivada
    gradDt = coords - Zt2;
    gradDtn = sqrt((meshgrid(gradDt(:,1))- meshgrid(gradDt(:,1)).').^2 + (meshgrid(gradDt(:,2))-meshgrid(gradDt(:,2)).').^2);
    %gradDtn = sqrt((meshgrid(Zt2(:,1))-meshgrid(Zt2(:,1)).').^2 + (meshgrid(Zt2(:,2))-meshgrid(Zt2(:,2)).').^2);
    X = X0 - t*gradDtn
    %a = X - X0;
    [Xevec,Xeval] = eig(X);
    
    %Projection
    for b = 1:size(X)
        for c = 1:size(X)
            if Xeval(b,c) <= 0
                Xeval(b,c) = 0;
            end %if
        end %c
    end %b
    
    %Compute the estimated coordinates
    [Yt2,eigvalst2] = cmdscale(abs(X),2);
    [Dt2,Zt2] = procrustes(coords,Yt2);
    %Plot estimated coordinates
    plot(Zt2(:,1),Zt2(:,2),'*',...
    'MarkerSize',5);
    X0 = X;
    it = it + 1
    elapsed = cputime - time
end %while
    
%Real coordinates
plot(coords(:,1),coords(:,2),'X',...
    'MarkerSize',10);
%Estimated coordinates with SDR matrix
plot(Zt1(:,1),Zt1(:,2),'o',...
    'MarkerSize',10);    