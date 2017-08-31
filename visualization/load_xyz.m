function [X] = load_xyz(filename)
%load_xyz Load point cloud from input file.
    X = dlmread(filename, ' ', 1, 0);
end