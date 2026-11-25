function [Q, metrics] = trainModelA(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams)
% TRAINMODELA  Trains the tabular Q-learning agent under MODEL A
% (baseline) state representation: state_A = [otherD2DChannel,
% QoSViolationCount], 32 states. This is the SAME algorithm and
% environment as Stage 11's trainQLearning.m; the only additions here
% are per-episode Q-table-change tracking and state/state-action
% visitation counting, needed for Stage 13's comparison.
%
% Reuses the unmodified wireless-environment functions from Stages 7-10
% (calculateSINR, calculateCellularSINR, calculateDataRate,
% calculateReward) and the unmodified selectAction.m / updateQTable.m
% from Stage 11.
%
% AGENT SCOPE (unchanged project simplification from Stage 11): the
% agent controls ONLY D2D pair 1. D2D pair 2's channel is redrawn
% uniformly at random every time step. This is NOT multi-agent
% reinforcement learning - it is tabular Q-learning-based channel
% selection for a single D2D transmitter in a cellular network, with
% the second D2D pair treated as part of the (non-stationary)
% environment.
%
% Inputs/Outputs: same conventions as Stage 11's trainQLearning.m, with
% metrics additionally including qChangeHistory, visitState,
% visitStateAction.

    numStates  = 32;
    numActions = 8;
    Q = zeros(numStates, numActions);

    visitState       = zeros(1, numStates);
    visitStateAction = zeros(numStates, numActions);

    numEpisodes  = hyperParams.numEpisodes;
    maxTimeSteps = hyperParams.maxTimeSteps;
    alpha        = hyperParams.alpha;
    gamma        = hyperParams.gamma;
    epsilon      = hyperParams.epsilon;
    epsilonMin   = hyperParams.epsilonMin;
    epsilonDecay = hyperParams.epsilonDecay;

    txPowerD2D    = envParams.txPowerD2D;
    P_CU          = envParams.P_CU;
    noisePower    = envParams.noisePower;
    W             = envParams.W;
    cuChannel     = envParams.cuChannel;
    R_min         = envParams.R_min;
    referenceRate = envParams.referenceRate;
    lambdaQoS     = envParams.lambdaQoS;
    numCU         = length(cuChannel);

    metrics.episodeReward     = zeros(1, numEpisodes);
    metrics.epsilonHistory    = zeros(1, numEpisodes);
    metrics.qChangeHistory    = zeros(1, numEpisodes);

    for episode = 1:numEpisodes

        Qprev = Q;   % snapshot at episode start, for Q-change tracking

        otherD2DChannel   = randi(numActions);
        qosViolationCount = 0;
        stateIndex = getStateModelA(otherD2DChannel, qosViolationCount);

        episodeRewardSum = 0;

        for t = 1:maxTimeSteps

            action = selectAction(Q, stateIndex, epsilon, numActions);
            [visitState, visitStateAction] = trackStateVisits(visitState, visitStateAction, stateIndex, action);

            A = [action, otherD2DChannel];

            d2dRate = zeros(1,2);
            for j = 1:2
                sinr_j = calculateSINR(txPowerD2D, H_D2D, A, j, noisePower);
                d2dRate(j) = calculateDataRate(sinr_j, W);
            end

            cuRate = zeros(1, numCU);
            for i = 1:numCU
                sinr_i = calculateCellularSINR(P_CU, h_CU_BS, txPowerD2D, h_D2D_BS, ...
                                                A, cuChannel, i, noisePower);
                cuRate(i) = calculateDataRate(sinr_i, W);
            end

            [reward, ~, ~, numViol] = calculateReward(d2dRate, cuRate, R_min, referenceRate, lambdaQoS);

            nextOtherD2DChannel   = randi(numActions);
            nextQosViolationCount = numViol;
            nextStateIndex = getStateModelA(nextOtherD2DChannel, nextQosViolationCount);

            Q = updateQTable(Q, stateIndex, action, reward, nextStateIndex, alpha, gamma);

            stateIndex      = nextStateIndex;
            otherD2DChannel = nextOtherD2DChannel;

            episodeRewardSum = episodeRewardSum + reward;

        end

        metrics.episodeReward(episode)  = episodeRewardSum;
        metrics.epsilonHistory(episode) = epsilon;
        metrics.qChangeHistory(episode) = trackQChange(Q, Qprev);

        epsilon = max(epsilonMin, epsilon * epsilonDecay);

    end

    metrics.visitState       = visitState;
    metrics.visitStateAction = visitStateAction;

end
