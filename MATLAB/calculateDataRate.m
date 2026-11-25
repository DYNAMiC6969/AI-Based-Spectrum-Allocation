function R = calculateDataRate(sinr, W)
% CALCULATEDATARATE  Shannon-capacity data rate from linear SINR.
%
%   R = W * log2(1 + SINR)
%
% Inputs:
%   sinr - linear (not dB) SINR value(s)
%   W    - channel bandwidth (our simulation assumption; use W = 1 to
%          get spectral efficiency in bits/s/Hz, which is what Stage 8's
%          test script uses)
%
% Output:
%   R    - data rate, same size as 'sinr'

    R = W .* log2(1 + sinr);

end
