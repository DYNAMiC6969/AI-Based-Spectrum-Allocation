function [Q, metrics] = trainModelB(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams)
% TRAINMODELB  Trains the tabular Q-learning agent under MODEL B
% (richer) state representation: state_B = [otherD2DChannel, q1, q2, q3],
% 64 states, where qi is CU_i's OWN binary QoS violation flag.
%
% Identical wireless environment, reward, and Q-learning mechanics to
% trainModelA.m - the ONLY difference is the state representation
% (getStateModelB instead of getStateModelA, and per-CU violation flags
% carried forward instead of a single violation count). This isolates
% the effect of state design, per Stage 13's purpose.
%
% Reuses the same unmodified wireless-environment and Q-learning
% mechanics functions as trainModelA.m.
%
% AGENT SCOPE: identical project simplification to trainModelA.m - only
% D2D pair 1 is controlled; D2D pair 2's channel is redrawn randomly
% every step. Not multi-agent RL.

    numStates  = 64;
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
    numCU         = length(cuChannel);   % = 3, matches q1,q2,q3

    metrics.episodeReward  = zeros(1, numEpisodes);
    metrics.epsilonHistory = zeros(1, numEpisodes);
    metrics.qChangeHistory = zeros(1, numEpisodes);

    for episode = 1:numEpisodes

        Qprev = Q;

        otherD2DChannel = randi(numActions);
        q1 = 0; q2 = 0; q3 = 0;   % initial assumption: no violations known yet
        stateIndex = getStateModelB(otherD2DChannel, q1, q2, q3);

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

            [reward, ~, ~, ~] = calculateReward(d2dRate, cuRate, R_min, referenceRate, lambdaQoS);

            % Per-CU violation flags (Model B's distinguishing information)
            nextQ1 = calculateQoSViolation(cuRate(1), R_min(1));
            nextQ2 = calculateQoSViolation(cuRate(2), R_min(2));
            nextQ3 = calculateQoSViolation(cuRate(3), R_min(3));

            nextOtherD2DChannel = randi(numActions);
            nextStateIndex = getStateModelB(nextOtherD2DChannel, nextQ1, nextQ2, nextQ3);

            Q = updateQTable(Q, stateIndex, action, reward, nextStateIndex, alpha, gamma);

            stateIndex      = nextStateIndex;
            otherD2DChannel = nextOtherD2DChannel;
            q1 = nextQ1; q2 = nextQ2; q3 = nextQ3;

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
