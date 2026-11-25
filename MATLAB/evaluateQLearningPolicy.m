function [avgD2DThroughput, avgCUThroughput, qosViolationRate, avgReward, QUnchanged] = ...
    evaluateQLearningPolicy(Q, H_D2D, h_CU_BS, h_D2D_BS, envParams, evaluationSteps, otherD2DChannelSequence)
% EVALUATEQLEARNINGPOLICY  Evaluates POLICY 2 (frozen trained Q-table) for
% D2D pair 1, over 'evaluationSteps' time steps.
%
% action = argmax(Q(state,:)) - GREEDY ONLY. No epsilon exploration, and
% the Q-table is NEVER updated during this evaluation (Q is only read
% via max(), never assigned back into).
%
% Ties in argmax are broken by MATLAB/Octave's default max() behavior
% (first occurrence), NOT randomized - this is a deliberate, documented
% choice for evaluation (unlike training's selectAction.m, which
% randomizes ties for exploration diversity; evaluation instead uses a
% fixed, reproducible tie-break).
%
% FAIRNESS: 'otherD2DChannel' at each step is taken from the SAME
% pre-generated 'otherD2DChannelSequence' passed to
% evaluateRandomPolicy.m for this run.
%
% qosViolationCount (part of the state) is carried forward causally from
% this policy's OWN previous-step outcome, exactly as in training
% (Stage 11) - it is NOT shared with the random policy's trajectory,
% since the random policy has no notion of state at all.
%
% Inputs: same as evaluateRandomPolicy.m, plus:
%   Q - [32x8] frozen, trained Q-table (Stage 11 output)
%
% Outputs: same four metrics as evaluateRandomPolicy.m, plus:
%   QUnchanged - true if Q is bit-for-bit identical to its input value
%                after this function returns (explicit verification,
%                even though MATLAB/Octave's pass-by-value semantics
%                make any other outcome structurally impossible here)

    Q_inputCopy = Q;   % kept only for the explicit unchanged-check below

    txPowerD2D    = envParams.txPowerD2D;
    P_CU          = envParams.P_CU;
    noisePower    = envParams.noisePower;
    W             = envParams.W;
    cuChannel     = envParams.cuChannel;
    R_min         = envParams.R_min;
    referenceRate = envParams.referenceRate;
    lambdaQoS     = envParams.lambdaQoS;

    numCU = length(cuChannel);

    qosViolationCount = 0;   % initial state assumption, same as training

    totalD2DRateSum = 0;
    totalCURateSum  = 0;
    totalQoSViol    = 0;
    totalReward     = 0;

    for t = 1:evaluationSteps

        otherD2DChannel = otherD2DChannelSequence(t);   % shared environment draw
        stateIndex = getState(otherD2DChannel, qosViolationCount);

        [~, action] = max(Q(stateIndex, :));   % POLICY 2: greedy only, no update

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

        qosViolationCount = numViol;   % causal carry-forward for next state

    end

    avgD2DThroughput = totalD2DRateSum / evaluationSteps;
    avgCUThroughput  = totalCURateSum / evaluationSteps;
    qosViolationRate = totalQoSViol / (numCU * evaluationSteps);
    avgReward        = totalReward / evaluationSteps;

    QUnchanged = isequal(Q, Q_inputCopy);

end
