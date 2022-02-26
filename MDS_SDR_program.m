clear all
close all
load('mds_train.mat');

% Formulating Data for running the Semi-Definite Relaxed Optimization
%D = transpose(rot90(time_matrix,2));
D = time_matrix.^2;
W = ones(size(time_matrix));
lambda = 0.6;

%Obtaining Coordintae Values
[EDM, X] = sdr_complete_edm_final(D, W, lambda);
%st = flip(station_index);
%c = flip(coords);
[Dt,Z] = procrustes(coords, X(1:2,:).');

%Error in Results
SDR_e = (sum(((Z-coords).^2.)').').^0.5;

tstart = tic;
% Obtaining Coordintae Values
[X] = GradDesc(D, W);
[L, M, N] = svd(X);
coo = sqrt(M)*N';
[Dt,Z1] = procrustes(coords, coo(1:2,:).');
time = toc(tstart);
% % Plotting the results
map_plot(coords,Z1,station_index);
