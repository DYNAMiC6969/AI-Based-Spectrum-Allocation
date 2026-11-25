function action = selectAction(Q, stateIndex, epsilon, numActions)
% SELECTACTION  Epsilon-greedy action selection with random tie-breaking.
%
% With probability epsilon: choose a uniformly random action (explore).
% Otherwise: choose the action with the maximum Q-value for this state
% (exploit). If multiple actions tie for the maximum, one is chosen
% uniformly at random among the tied actions.
%
% Inputs:
%   Q           - [numStates x numActions] Q-table
%   stateIndex  - current state index (row into Q)
%   epsilon     - current exploration probability
%   numActions  - number of available actions (= 8, the channels)
%
% Output:
%   action      - selected action index in {1,...,numActions}

    if rand() < epsilon
        action = randi(numActions);
    else
        qRow = Q(stateIndex, :);
        maxQ = max(qRow);
        bestActions = find(qRow == maxQ);
        action = bestActions(randi(length(bestActions)));
    end

end
