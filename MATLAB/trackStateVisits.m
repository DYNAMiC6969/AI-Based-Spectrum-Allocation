function [visitState, visitStateAction] = trackStateVisits(visitState, visitStateAction, stateIndex, action)
% TRACKSTATEVISITS  Increments visitation counters for the given state
% and state-action pair.
%
% Inputs:
%   visitState        - [1 x numStates] running visit count per state
%   visitStateAction  - [numStates x numActions] running visit count
%                        per state-action pair
%   stateIndex         - state visited this step
%   action             - action taken this step
%
% Outputs:
%   visitState, visitStateAction - updated counters

    visitState(stateIndex) = visitState(stateIndex) + 1;
    visitStateAction(stateIndex, action) = visitStateAction(stateIndex, action) + 1;

end
