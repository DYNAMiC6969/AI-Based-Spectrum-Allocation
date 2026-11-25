function stateIndex = getState(otherD2DChannel, qosViolationCount)
% GETSTATE  Maps the compact state [otherD2DChannel, QoSViolationCount]
% to a single integer state index for the Q-table.
%
% STATE DEFINITION (project simulation assumption, Stage 11):
%   otherD2DChannel   in {1,...,8}  - channel currently used by the D2D
%                                     pair NOT controlled by this agent
%   qosViolationCount in {0,1,2,3}  - number of cellular users (out of 3)
%                                     currently below their QoS threshold
%
% Total states = 8 x 4 = 32.
%
% Mapping (documented, deterministic, no gaps/overlaps):
%   stateIndex = (otherD2DChannel - 1) * 4 + qosViolationCount + 1
%
% Examples:
%   otherD2DChannel=1, qosViolationCount=0 -> stateIndex = 1
%   otherD2DChannel=1, qosViolationCount=3 -> stateIndex = 4
%   otherD2DChannel=2, qosViolationCount=0 -> stateIndex = 5
%   otherD2DChannel=8, qosViolationCount=3 -> stateIndex = 32
%
% Inputs:
%   otherD2DChannel   - integer in {1,...,8}
%   qosViolationCount - integer in {0,1,2,3}
%
% Output:
%   stateIndex - integer in {1,...,32}

    stateIndex = (otherD2DChannel - 1) * 4 + qosViolationCount + 1;

end
