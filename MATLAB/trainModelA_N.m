function [Q, metrics] = trainModelA_N(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams, numChannels)
% TRAINMODELA_N  Generalized version of Stage 11/13's trainModelA.m,
% parameterized by 'numChannels' instead of a hardcoded 8. Used by
% Stage 14's Experiment 1 (numChannels=8, matching the original
% baseline exactly) and Experiment 3 (numChannels in {4,6,8,10}).
%
% STATE MODEL: Model A (selected as sufficient in Stage 13):
%   state_A = [otherD2DChannel, QoSViolationCount]
%   otherD2DChannel in {1,...,numChannels}, QoSViolationCount in {0,1,2,3}
%   numStates = numChannels * 4  (matches Stage 14's Experiment 3 formula)
%
% ACTION SPACE: A = {1,...,numChannels}  (D2D pair 1 only - the agent)
%
% Reuses the SAME wireless-environment and reward functions as every
% prior stage, unmodified (calculateSINR, calculateCellularSINR,
% calculateDataRate, calculateReward), and the SAME getState.m indexing
% formula (which is agnostic to numChannels - only the valid input
% range and the resulting Q-table size change).
%
% AGENT SCOPE (unchanged project simplification): only D2D pair 1 is
% controlled. D2D pair 2's channel is redrawn uniformly at random from
% {1,...,numChannels} every time step. NOT multi-agent RL.
%
% Inputs:
%   net, H_D2D, h_CU_BS, h_D2D_BS - environment/channel-gain data
%   envParams   - struct: txPowerD2D, P_CU, noisePower, W, cuChannel,
%                 R_min, referenceRate, lambdaQoS
%   hyperParams - struct: alpha, gamma, epsilon, epsilonMin,
%                 epsilonDecay, numEpisodes, maxTimeSteps
%   numChannels - number of available channels (action space size)
%
% Outputs:
%   Q       - [numChannels*4 x numChannels] learned Q-table
%   metrics - struct (per-episode arrays; DEFINITIONS match the evaluation
%             functions so training and evaluation numbers are comparable):
%               episodeReward        - episode RETURN: sum of rewards over the episode
%               averageReward        - episodeReward / maxTimeSteps (per-step mean; same
%                                      definition as evaluation 'avgReward')
%               averageD2DThroughput - mean over steps of AGGREGATE D2D rate (R_D2D1+R_D2D2)
%               averageCellularRate  - mean over steps of the mean-over-CUs cellular rate
%               QoSViolationRate     - sum(N_QoS) / (numCU * maxTimeSteps)
%               epsilonHistory

    numStates  = numChannels * 4;
    numActions = numChannels;
    Q = zeros(numStates, numActions);

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

    metrics.episodeReward       = zeros(1, numEpisodes);
    metrics.averageReward       = zeros(1, numEpisodes);
    metrics.averageD2DThroughput = zeros(1, numEpisodes);
    metrics.averageCellularRate = zeros(1, numEpisodes);
    metrics.QoSViolationRate    = zeros(1, numEpisodes);
    metrics.epsilonHistory      = zeros(1, numEpisodes);

    for episode = 1:numEpisodes

        otherD2DChannel   = randi(numActions);
        qosViolationCount = 0;
        stateIndex = getState(otherD2DChannel, qosViolationCount);

        episodeRewardSum  = 0;
        episodeD2DRateSum = 0;
        episodeCURateSum  = 0;
        episodeQoSSum     = 0;

        for t = 1:maxTimeSteps

            action = selectAction(Q, stateIndex, epsilon, numActions);

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
            nextStateIndex = getState(nextOtherD2DChannel, nextQosViolationCount);

            Q = updateQTable(Q, stateIndex, action, reward, nextStateIndex, alpha, gamma);

            stateIndex      = nextStateIndex;
            otherD2DChannel = nextOtherD2DChannel;

            episodeRewardSum  = episodeRewardSum + reward;
            episodeD2DRateSum = episodeD2DRateSum + sum(d2dRate);
            episodeCURateSum  = episodeCURateSum + mean(cuRate);
            episodeQoSSum     = episodeQoSSum + numViol;

        end

        metrics.episodeReward(episode)       = episodeRewardSum;
        metrics.averageReward(episode)       = episodeRewardSum / maxTimeSteps;
        metrics.averageD2DThroughput(episode) = episodeD2DRateSum / maxTimeSteps;
        metrics.averageCellularRate(episode) = episodeCURateSum / maxTimeSteps;
        metrics.QoSViolationRate(episode)    = episodeQoSSum / (numCU * maxTimeSteps);
        metrics.epsilonHistory(episode)      = epsilon;

        epsilon = max(epsilonMin, epsilon * epsilonDecay);

    end

end
