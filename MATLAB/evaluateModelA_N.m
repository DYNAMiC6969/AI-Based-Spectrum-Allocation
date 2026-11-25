
function [avgD2DThroughput, avgCUThroughput, ...
          qosViolationRate, avgReward, QUnchanged] = ...
    evaluateModelA_N(Q, H_D2D, h_CU_BS, h_D2D_BS, ...
                     envParams, evaluationSteps, ...
                     otherD2DChannelSequence, numChannels)
% EVALUATEMODELA_N
% Evaluates a frozen Model A Q-table for a specified channel count.
% No exploration and no Q-table updates occur during evaluation.

    if numChannels < 1 || numChannels ~= floor(numChannels)
        error('numChannels must be a positive integer.');
    end

    if size(Q,1) ~= 4*numChannels || size(Q,2) ~= numChannels
        error('Q-table dimensions do not match numChannels.');
    end

    if numel(otherD2DChannelSequence) ~= evaluationSteps
        error('Channel sequence length must equal evaluationSteps.');
    end

    if any(otherD2DChannelSequence < 1) || ...
       any(otherD2DChannelSequence > numChannels) || ...
       any(otherD2DChannelSequence ~= floor(otherD2DChannelSequence))
        error('Other-pair channels must be integers from 1 to numChannels.');
    end

    Q_original = Q;

    txPowerD2D    = envParams.txPowerD2D;
    P_CU          = envParams.P_CU;
    noisePower    = envParams.noisePower;
    W             = envParams.W;
    cuChannel     = envParams.cuChannel;
    R_min         = envParams.R_min;
    referenceRate = envParams.referenceRate;
    lambdaQoS     = envParams.lambdaQoS;

    numCU = length(cuChannel);
    qosViolationCount = 0;

    totalD2D = 0;
    totalCU = 0;
    totalQoSViolations = 0;
    totalReward = 0;

    for t = 1:evaluationSteps

        otherChannel = otherD2DChannelSequence(t);

        stateIndex = getState(otherChannel, qosViolationCount);

        % Greedy action selection; Q remains frozen.
        [~, action] = max(Q(stateIndex, :));

        A = [action, otherChannel];

        d2dRate = zeros(1,2);
        for j = 1:2
            sinr_j = calculateSINR( ...
                txPowerD2D, H_D2D, A, j, noisePower);
            d2dRate(j) = calculateDataRate(sinr_j, W);
        end

        cuRate = zeros(1,numCU);
        for i = 1:numCU
            sinr_i = calculateCellularSINR( ...
                P_CU, h_CU_BS, txPowerD2D, ...
                h_D2D_BS, A, cuChannel, i, noisePower);
            cuRate(i) = calculateDataRate(sinr_i, W);
        end

        [reward, totalD2DRateStep, ~, numViol] = ...
            calculateReward( ...
                d2dRate, cuRate, R_min, ...
                referenceRate, lambdaQoS);

        totalD2D = totalD2D + totalD2DRateStep;
        totalCU = totalCU + mean(cuRate);
        totalQoSViolations = totalQoSViolations + numViol;
        totalReward = totalReward + reward;

        % Carry this policy's own QoS outcome into its next state.
        qosViolationCount = numViol;
    end

    avgD2DThroughput = totalD2D / evaluationSteps;
    avgCUThroughput = totalCU / evaluationSteps;
    qosViolationRate = totalQoSViolations / ...
                       (numCU * evaluationSteps);
    avgReward = totalReward / evaluationSteps;

    QUnchanged = isequal(Q, Q_original);
end
