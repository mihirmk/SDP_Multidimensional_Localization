load('mds_train.mat')
set(gca, 'YDir','reverse')
scatter(coords(:,1),coords(:,2))
dx = 0.1; dy = 0.0; % displacement so the text does not overlay the data points
text(coords(:,1)+dx, coords(:,2)+dy, station_index);
estimated_coordinates_dist = cmdscale(distance);
estimated_coordinates_time = cmdscale(time_matrix,2);
[Dt,Z] = procrustes(coords, estimated_coordinates_time);%time
%[Dt,Z] = procrustes(coords, estimated_coordinates_dist);%distance

%Error in Results
CMD_e = (sum(((Z-coords).^2.)').').^0.5; %error vector of predicted loaction and actual location

map_plot(coords,Z,station_index)

function p = map_plot (actual_coords, estimated_coords,station_index)
set(gca, 'YDir','reverse')
p = plot(actual_coords(:,1),actual_coords(:,2),'b.',estimated_coords(:,1),estimated_coords(:,2),'rx');
dx = 0.1; dy = 0.1;
text(actual_coords(:,1)+dx, actual_coords(:,2)+dy, station_index);
dy=0.1;
text(estimated_coords(:,1)+dx, estimated_coords(:,2)+dy, station_index);
end
