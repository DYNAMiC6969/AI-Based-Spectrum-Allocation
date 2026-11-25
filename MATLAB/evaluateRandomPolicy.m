function [avgD2DThroughput, avgCUThroughput, qosViolationRate, avgReward] = ...
    evaluateRandomPolicy(H_D2D, h_CU_BS, h_D2D_BS, envParams, evaluationSteps, otherD2DChannelSequence)
% EVALUATERANDOMPOLICY  Evaluates POLICY 1 (random channel selection) for
% D2D pair 1, over 'evaluationSteps' time steps.
%
% No learning, no Q-table, no epsilon-greedy - action is drawn uniformly
% at random from {1,...,8} at every step, as specified for Step 12.
%
% FAIRNESS: 'otherD2DChannel' at each step is taken from the PRE-GENERATED
% 'otherD2DChannelSequence' (same sequence also passed to
% evaluateQLearningPolicy.m for this run), rather than drawn fresh here -
% this is what keeps the environment identical across both policies
% within one evaluation run.
%
% Inputs:
%   H_D2D, h_CU_BS, h_D2D_BS - channel-gain matrices (Stages 7,9)
%   envParams                - struct: txPowerD2D, P_CU, noisePower, W,
%                               cuChannel, R_min, referenceRate, lambdaQoS
%   evaluationSteps           - number of time steps to evaluate
%   otherD2DChannelSequence   - [1 x evaluationSteps] pre-generated
%                               channel sequence for D2D pair 2
%
% Outputs (averaged over evaluationSteps):
%   avgD2DThroughput  - mean aggregate D2D rate (R_D2D1+R_D2D2) per step
%   avgCUThroughput   - mean average cellular rate per step
%   qosViolationRate  - sum(N_QoS) / (numCU * evaluationSteps)
%   avgReward         - mean reward per step

    txPowerD2D    = envParams.txPowerD2D;
    P_CU          = envParams.P_CU;
    noisePower    = envParams.noisePower;
    W             = envParams.W;
    cuChannel     = envParams.cuChannel;
    R_min         = envParams.R_min;
    referenceRate = envParams.referenceRate;
    lambdaQoS     = envParams.lambdaQoS;

    numCU = length(cuChannel);
    numActions = 8;

    totalD2DRateSum = 0;
    totalCURateSum  = 0;
    totalQoSViol    = 0;
    totalReward     = 0;

    for t = 1:evaluationSteps

        action = randi(numActions);                 % POLICY 1: pure random
        otherD2DChannel = otherD2DChannelSequence(t); % shared environment draw

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

        [reward, totalD2DRateStep, ~, numViol] = ...
            calculateReward(d2dRate, cuRate, R_min, referenceRate, lambdaQoS);

        totalD2DRateSum = totalD2DRateSum + totalD2DRateStep;
        totalCURateSum  = totalCURateSum + mean(cuRate);
        totalQoSViol    = totalQoSViol + numViol;
        totalReward     = totalReward + reward;

    end

    avgD2DThroughput = totalD2DRateSum / evaluationSteps;
    avgCUThroughput  = totalCURateSum / evaluationSteps;
    qosViolationRate = totalQoSViol / (numCU * evaluationSteps);
    avgReward        = totalReward / evaluationSteps;

end
