function [T1, T2] = ods_project(X, C, r)
%ods_project Project x into camera with centre c and interpupillary
% distance r.
    x = X(1, :) - C(1, :);
    y = X(2, :) - C(2, :);
    z = X(3, :) - C(3, :);
    
    perp_check = @(t) abs(x .* cos(t) + z .* sin(t) - r) < 1E-6;
    
    % Horizontal headings.
    T1(1, :) = pi + asin(r ./ sqrt(x .^2 + z .^ 2)) - atan(x ./ z);
    T2(1, :) = asin(- r ./ sqrt(x .^2 + z .^ 2)) - atan(x ./ z);
    T1(1, ~perp_check(T1(1, :))) = T1(1, ~perp_check(T1(1, :))) + pi;
    T2(1, ~perp_check(T2(1, :))) = T2(1, ~perp_check(T2(1, :))) + pi;
    
    C_P = @(t) [r * cos(t); zeros(1, size(t, 2)); r * sin(t)];
    U1 = X - C_P(T1(1, :));
    U2 = X - C_P(T2(1, :));
    
    V1 = U1;
    V2 = U2;
    V1(2, :) = 0;
    V2(2, :) = 0;
    
    % Vertical headings.
    T1(2, :) = acos(dot(U1, V1) ./ sqrt(sum(U1 .^ 2, 1) .* sum(V1 .^ 2, 1)));
    T2(2, :) = acos(dot(U2, V2) ./ sqrt(sum(U2 .^ 2, 1) .* sum(V2 .^ 2, 1)));
end