function [r_opt] = optimise(a, b)
    % Least squares loss on horizontal disparity.
    lambda_loss = @(r) loss(a, b, r);

    % Simple grid search for coarse optimisation.
    nx = 9;
    ny = 9;
    nz = 9;
    values = zeros(nx, ny, nz, 3);
    losses = zeros(nx, ny, nz);
    i = 1;
    for x = linspace(-0.07, 0.07, nx)
        j = 1;
        for y = linspace(-0.07, 0.07, ny)
            k = 1;
            for z = linspace(-0.07, 0.07, nz)
                values(i, j, k, :) = [x, y, z];
                losses(i, j, k) = lambda_loss([x, y, z]);
                fprintf('Loss at (%1.2f, %1.2f, %1.2f) : %1.5f\n', x, y, z, losses(i, j, k));
                k = k + 1;
            end
            j = j + 1;
        end
        i = i + 1;
    end
    
    [min_val, idx] = min(losses(:));
    disp(min_val);
    [i, j, k] = ind2sub(size(losses), idx);
    r_initial = squeeze(values(i, j, k, :));
    
    % Refine with nonlinear optimisation.
    opt = optimset('Display', 'iter', 'MaxFunEvals', 32, 'MaxIter', 32);
    r_opt = fminsearch(@(r) lambda_loss(r) + 0.5 * sum((r_initial - r) .^ 2), r_initial, opt);
end