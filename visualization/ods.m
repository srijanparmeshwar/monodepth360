function [Y] = ods(X, output_size)
    h = output_size(1);
    w = output_size(2);
    [S_L, T_L, S_R, T_R] = ods_project(X(:, 1:3), 0.065);
    
    u_L = (pi + S_L) / (2 * pi);
    v_L = (pi / 2 + T_L) / pi;
    
    u_R = (pi + S_R) / (2 * pi);
    v_R = (pi / 2 + T_R) / pi;
    
    Y_L = zeros(h, w, 3);
    Y_R = zeros(h, w, 3);
    [u, v] = meshgrid(linspace(0, 1, w), linspace(0, 1, h));
    for channel = 1:3
        channel_L = griddata(u_L, v_L, X(:, 3 + channel), u(:), v(:), 'linear');
        channel_R = griddata(u_R, v_R, X(:, 3 + channel), u(:), v(:), 'linear');
        Y_L(:, :, channel) = reshape(channel_L, [h, w]);
        Y_R(:, :, channel) = reshape(channel_R, [h, w]);
    end
    
    Y = cat(1, Y_L, Y_R);
end