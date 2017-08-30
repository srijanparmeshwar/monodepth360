function [X] = load_xyz(filename)
    X = dlmread(filename, ' ', 1, 0);
end