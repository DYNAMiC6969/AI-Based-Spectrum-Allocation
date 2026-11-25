function [avgD2DThroughput, avgCUThroughput, qosViolationRate, avgReward, QUnchanged] = ...
    evaluateModelBPolicy(Q, H_D2D, h_CU_BS, h_D2D_BS, envParams, evaluationSteps, otherD2DChannelSequence)
% EVALUATEMODELBPOLICY  Evaluates a frozen, trained Model B (64-state)
% Q-learning policy - greedy only (argmax), no exploration, no updates.
% Directly analogous to Stage 12's evaluateQLearningPolicy.m (Model A),
% but tracks per-CU violation flags (q1,q2,q3) instead of a single
% violation count, matching getStateModelB.m's state definition.
%
% See evaluateQLearningPolicy.m for the full rationale on frozen-Q
% verification and shared-sequence fairness; identical conventions apply
% here.

    Q_inputCopy = Q;

    txPowerD2D    = envParams.txPowerD2D;
    P_CU          = envParams.P_CU;
    noisePower    = envParams.noisePower;
    W             = envParams.W;
    cuChannel     = envParams.cuChannel;
    R_min         = envParams.R_min;
    referenceRate = envParams.referenceRate;
    lambdaQoS     = envParams.lambdaQoS;

    numCU = length(cuChannel);

    q1 = 0; q2 = 0; q3 = 0;   % initial assumption, same as training

    totalD2DRateSum = 0;
    totalCURateSum  = 0;
    totalQoSViol    = 0;
    totalReward     = 0;

    for t = 1:evaluationSteps

        otherD2DChannel = otherD2DChannelSequence(t);
        stateIndex = getStateModelB(otherD2DChannel, q1, q2, q3);

        [~, action] = max(Q(stateIndex, :));

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

        q1 = calculateQoSViolation(cuRate(1), R_min(1));
        q2 = calculateQoSViolation(cuRate(2), R_min(2));
        q3 = calculateQoSViolation(cuRate(3), R_min(3));

    end

    avgD2DThroughput = totalD2DRateSum / evaluationSteps;
    avgCUThroughput  = totalCURateSum / evaluationSteps;
    qosViolationRate = totalQoSViol / (numCU * evaluationSteps);
    avgReward        = totalReward / evaluationSteps;

    QUnchanged = isequal(Q, Q_inputCopy);

end
