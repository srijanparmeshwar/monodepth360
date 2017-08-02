function [S, T] = lat_long_grid(w, h)
%lat_long_grid Create grid of latitude and longitude coordinates for an
% equirectangular image.
    [S, T] = meshgrid(linspace(-pi + 1e-6, pi - 1e-6, w), linspace(-pi / 2 + 1e-6, pi / 2 - 1e-6, h));
end