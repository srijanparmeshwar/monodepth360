% Create grid.
[X, Y] = meshgrid(linspace(0, 1, 512));

% Rectangular function.
rect = @(x) 0.5 * (sign(x + 0.5) - sign(x - 0.5));

% Box function.
box = @(X, Y, a, b, s) rect(s * (X - a)) .* rect(s * (Y - b));

% Discrete Laplacian kernel.
kernel = [
            0, -1, 0;
            -1, 4, -1;
            0, -1, 0
         ];
       
% Draw boxes.  
original = box(X, Y, 0.25, 0.25, 4);
translated = box(X, Y, 0.75, 0.75, 4);

% Apply filter.
original_filtered = conv2(original, kernel, 'same');
translated_filtered = conv2(translated, kernel, 'same');

% Save figures.
imwrite(original, 'original.jpg');
imwrite(translated, 'translated.jpg');
imwrite(original_filtered, 'original_filtered.jpg');
imwrite(translated_filtered, 'translated_filtered.jpg');
