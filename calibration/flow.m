function [vx, vy] = flow(a, b)
    alpha = 0.012;
    ratio = 0.75;
    minWidth = 20;
    nOuterFPIterations = 6;
    nInnerFPIterations = 1;
    nSORIterations = 25;

    para = [alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations];

    [vx, vy] = Coarse2FineTwoFrames(a, b, para);
end

