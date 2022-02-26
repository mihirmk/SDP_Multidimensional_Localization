function p = map_plot (actual_coords, estimated_coords,station_index)
set(gca, 'YDir','reverse')
p = plot(actual_coords(:,1),actual_coords(:,2),'b.',estimated_coords(:,1),estimated_coords(:,2),'rx');
dx = 0.1; dy = 0.1;
text(actual_coords(:,1)+dx, actual_coords(:,2)+dy, station_index);
dy=0.1;
text(estimated_coords(:,1)+dx, estimated_coords(:,2)+dy, station_index);
end