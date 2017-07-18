function [S, T] = lat_long_grid(w, h)
%lat_long_grid Create grid of latitude and longitude coordinates for an
% equirectangular image.
    [S, T] = meshgrid(linspace(-pi, pi, w), linspace(-pi / 2, pi / 2, h));
end