function [loss] = loss(a, b, r)
    [vx, ~] = flow(rotate(a, r), b);
    [h, w] = size(vx);
    loss = sum(...
        ((reshape(vx((h / 4):(3 * h / 4), (w / 4):(3 * w / 4)), 1, []) / w) .^ 2)...
    ) + 0.5 * (r(1) ^ 2 + r(2) ^ 2);
end