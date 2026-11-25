function [Q, metrics] = trainQLearning(net, H_D2D, h_CU_BS, h_D2D_BS, envParams, hyperParams)
% TRAINQLEARNING  Tabular Q-learning training loop for single-D2D-pair
% channel allocation, under the corrected cellular UPLINK model
% (CU -> BS desired signal, D2D TX -> BS interference).
%
% AGENT SCOPE (project simulation assumption, Stage 11):
%   The agent controls ONLY D2D pair 1's channel selection. D2D pair 2
%   ("the other D2D pair") is NOT controlled by this agent; its channel
%   is redrawn uniformly at random at every time step, representing its
%   autonomous, unmodeled behavior. This is what makes
%   "otherD2DChannel" a meaningful, time-varying part of the state - see
%   the Stage 11 document, Section 2, for the full justification.
%
% Inputs:
%   net         - struct from generatePositions() (positions, simArea)
%   H_D2D       - [2x2] D2D-to-D2D channel gain matrix (Stage 7)
%   h_CU_BS     - [3x1] CU-to-BS channel gain vector (Stage 9, uplink)
%   h_D2D_BS    - [2x1] D2D-transmitter-to-BS channel gain vector (Stage 9)
%   envParams   - struct with fields:
%                   txPowerD2D, P_CU, noisePower, W, cuChannel, R_min,
%                   referenceRate, lambdaQoS
%   hyperParams - struct with fields:
%                   alpha, gamma, epsilon, epsilonMin, epsilonDecay,
%                   numEpisodes, maxTimeSteps
%
% Outputs:
%   Q       - [32x8] learned Q-table
%   metrics - struct with per-episode arrays:
%               episodeReward (episode return = sum over steps), averageReward (per-step mean),
%               averageD2DThroughput (mean over steps of AGGREGATE R_D2D1+R_D2D2),
%               averageCellularRate (mean over CUs), QoSViolationRate
%               (= sum(N_QoS)/(numCU*maxTimeSteps)), epsilonHistory,
%               averageD2DSINR, averageCellularSINR

    numStates  = 32;
    numActions = 8;
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

    metrics.episodeReward        = zeros(1, numEpisodes);
    metrics.averageReward        = zeros(1, numEpisodes);
    metrics.averageD2DThroughput = zeros(1, numEpisodes);
    metrics.averageCellularRate  = zeros(1, numEpisodes);
    metrics.QoSViolationRate     = zeros(1, numEpisodes);
    metrics.epsilonHistory       = zeros(1, numEpisodes);
    metrics.averageD2DSINR       = zeros(1, numEpisodes);
    metrics.averageCellularSINR  = zeros(1, numEpisodes);

    for episode = 1:numEpisodes

        % 1-2. Reset environment / initialize state
        otherD2DChannel   = randi(numActions);
        qosViolationCount = 0;
        stateIndex = getState(otherD2DChannel, qosViolationCount);

        episodeRewardSum   = 0;
        episodeD2DRateSum  = 0;
        episodeCURateSum   = 0;
        episodeQoSSum      = 0;
        episodeD2DSINRSum  = 0;
        episodeCUSINRSum   = 0;

        for t = 1:maxTimeSteps

            % a. Select action (epsilon-greedy)
            action = selectAction(Q, stateIndex, epsilon, numActions);

            % b. Apply the channel-selection action
            A = [action, otherD2DChannel];   % D2D pair 1 = agent, pair 2 = other

            % c-e. D2D interference, SINR, data rate
            d2dSINR  = zeros(1,2);
            d2dRate  = zeros(1,2);
            for j = 1:2
                d2dSINR(j) = calculateSINR(txPowerD2D, H_D2D, A, j, noisePower);
                d2dRate(j) = calculateDataRate(d2dSINR(j), W);
            end

            % f-h. Cellular uplink interference, SINR, data rate
            numCU = length(cuChannel);
            cuSINR = zeros(1, numCU);
            cuRate = zeros(1, numCU);
            for i = 1:numCU
                cuSINR(i) = calculateCellularSINR(P_CU, h_CU_BS, txPowerD2D, h_D2D_BS, ...
                                                   A, cuChannel, i, noisePower);
                cuRate(i) = calculateDataRate(cuSINR(i), W);
            end

            % i-j. QoS violations and reward
            [reward, ~, ~, numberOfQoSViolations] = ...
                calculateReward(d2dRate, cuRate, R_min, referenceRate, lambdaQoS);

            % k. Determine next state:
            %    - otherD2DChannel is redrawn (models pair 2's autonomous move)
            %    - qosViolationCount carries forward THIS step's outcome
            nextOtherD2DChannel   = randi(numActions);
            nextQosViolationCount = numberOfQoSViolations;
            nextStateIndex = getState(nextOtherD2DChannel, nextQosViolationCount);

            % l. Update Q-table
            Q = updateQTable(Q, stateIndex, action, reward, nextStateIndex, alpha, gamma);

            % m. Move to next state
            stateIndex      = nextStateIndex;
            otherD2DChannel = nextOtherD2DChannel;

            % Accumulate per-step metrics
            episodeRewardSum  = episodeRewardSum + reward;
            episodeD2DRateSum = episodeD2DRateSum + sum(d2dRate);
            episodeCURateSum  = episodeCURateSum + mean(cuRate);
            episodeQoSSum     = episodeQoSSum + numberOfQoSViolations;
            episodeD2DSINRSum = episodeD2DSINRSum + mean(d2dSINR);
            episodeCUSINRSum  = episodeCUSINRSum + mean(cuSINR);

        end

        % 4-5. Store episode-level metrics (averaged over time steps)
        metrics.episodeReward(episode)       = episodeRewardSum;
        metrics.averageReward(episode)       = episodeRewardSum / maxTimeSteps;
        metrics.averageD2DThroughput(episode) = episodeD2DRateSum / maxTimeSteps;
        metrics.averageCellularRate(episode) = episodeCURateSum / maxTimeSteps;
        metrics.QoSViolationRate(episode)    = episodeQoSSum / (length(cuChannel) * maxTimeSteps);
        metrics.epsilonHistory(episode)      = epsilon;
        metrics.averageD2DSINR(episode)      = episodeD2DSINRSum / maxTimeSteps;
        metrics.averageCellularSINR(episode) = episodeCUSINRSum / maxTimeSteps;

        % 6. Decay epsilon
        epsilon = max(epsilonMin, epsilon * epsilonDecay);

    end

end
