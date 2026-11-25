function qChange = trackQChange(Qcurrent, Qprevious)
% TRACKQCHANGE  Magnitude of change in the Q-table between two snapshots.
%
%   qChange = norm(Qcurrent(:) - Qprevious(:))
%
% Used once per episode (Qprevious = Q-table snapshot taken at the start
% of the episode, Qcurrent = Q-table after that episode's updates) to
% produce an "Episode vs Q-table change" stability curve: a qChange
% trending toward 0 suggests the Q-values are converging/stabilizing.
%
% Inputs:
%   Qcurrent, Qprevious - Q-tables of identical size
%
% Output:
%   qChange - nonnegative scalar (Euclidean/Frobenius norm of the
%             element-wise difference)

    qChange = norm(Qcurrent(:) - Qprevious(:));

end
