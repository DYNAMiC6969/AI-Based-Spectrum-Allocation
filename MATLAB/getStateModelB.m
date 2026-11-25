function stateIndex = getStateModelB(otherD2DChannel, q1, q2, q3)
% GETSTATEMODELB  MODEL B (richer) state indexing.
%
% state_B = [otherD2DChannel, q1, q2, q3] where qi in {0,1} is CU_i's
% OWN QoS violation flag (0 = satisfies R_min_i, 1 = violates it).
%
% This distinguishes WHICH cellular user(s) are violated, unlike Model A
% which only tracks the total COUNT of violations - the central
% experimental question this stage investigates.
%
% Total states = 8 (otherD2DChannel) x 2^3 (q1,q2,q3 combinations) = 64.
%
% Mapping (documented, deterministic, no gaps/overlaps):
%   qIndex     = q1*4 + q2*2 + q3 + 1          -> integer in {1,...,8}
%   stateIndex = (otherD2DChannel - 1) * 8 + qIndex   -> integer in {1,...,64}
%
% Examples:
%   otherD2DChannel=1, [q1 q2 q3]=[0 0 0] -> qIndex=1 -> stateIndex=1
%   otherD2DChannel=1, [q1 q2 q3]=[1 1 1] -> qIndex=8 -> stateIndex=8
%   otherD2DChannel=8, [q1 q2 q3]=[1 1 1] -> qIndex=8 -> stateIndex=64
%
% Inputs:
%   otherD2DChannel - integer in {1,...,8}
%   q1, q2, q3      - each 0 or 1 (QoS violation flag for CU1, CU2, CU3)
%
% Output:
%   stateIndex - integer in {1,...,64}

    qIndex = q1 * 4 + q2 * 2 + q3 + 1;
    stateIndex = (otherD2DChannel - 1) * 8 + qIndex;

end
