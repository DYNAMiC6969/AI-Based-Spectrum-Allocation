function d = calculateDistance(p1, p2)
% CALCULATEDISTANCE  Euclidean distance between two 2D points.
%
% Inputs:
%   p1 - [x1 y1] coordinate
%   p2 - [x2 y2] coordinate
%
% Output:
%   d  - Euclidean distance between p1 and p2 (meters)
%
% Formula:
%   d = sqrt( (x1-x2)^2 + (y1-y2)^2 )

    d = sqrt((p1(1) - p2(1))^2 + (p1(2) - p2(2))^2);

end
