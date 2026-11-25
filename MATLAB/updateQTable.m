function Q = updateQTable(Q, stateIndex, action, reward, nextStateIndex, alpha, gamma)
% UPDATEQTABLE  Standard tabular Q-learning update.
%
%   Q(s,a) <- Q(s,a) + alpha * (reward + gamma*max(Q(s',:)) - Q(s,a))
%
% Inputs:
%   Q              - [numStates x numActions] Q-table
%   stateIndex     - state s at the time of the action
%   action         - action a taken
%   reward         - reward received
%   nextStateIndex - resulting next state s'
%   alpha          - learning rate (simulation hyperparameter)
%   gamma          - discount factor (simulation hyperparameter)
%
% Output:
%   Q              - updated Q-table

    bestNextQ = max(Q(nextStateIndex, :));

    Q(stateIndex, action) = Q(stateIndex, action) + ...
        alpha * (reward + gamma * bestNextQ - Q(stateIndex, action));

end
