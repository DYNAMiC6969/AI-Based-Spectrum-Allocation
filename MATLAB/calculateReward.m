function [reward, totalD2DRate, normalizedD2DRate, numberOfQoSViolations] = ...
    calculateReward(d2dRates, cuRates, minimumCURates, referenceRate, lambdaQoS)
% CALCULATEREWARD  Stage 10 reward function.
%
%   reward = (sum(d2dRates) / referenceRate) - lambdaQoS * N_QoS
%
% This is a PROJECT DESIGN CHOICE, not a formula taken from the
% professor-provided thesis. No separate interference penalty is
% included: interference already shapes d2dRates (via D2D-to-D2D
% interference -> SINR -> rate) and cuRates (via D2D-to-BS interference
% -> uplink SINR -> rate), so an additional interference term would
% double-count the same physical effect the reward already responds to.
%
% Inputs:
%   d2dRates       - [1xJ] D2D data rates (e.g. from calculateDataRate.m)
%   cuRates        - [1xI] cellular uplink data rates
%   minimumCURates - [1xI] minimum acceptable rate per CU (R_min)
%   referenceRate  - scalar normalization constant for D2D throughput
%                    (project simulation assumption, NOT thesis-derived)
%   lambdaQoS      - scalar penalty weight per QoS violation
%                    (project simulation assumption, NOT thesis-derived)
%
% Outputs:
%   reward                 - scalar reward value
%   totalD2DRate           - sum(d2dRates), for diagnostics
%   normalizedD2DRate      - totalD2DRate / referenceRate, for diagnostics
%   numberOfQoSViolations  - N_QoS, for diagnostics

    totalD2DRate = sum(d2dRates);
    normalizedD2DRate = totalD2DRate / referenceRate;

    violations = cuRates < minimumCURates;
    numberOfQoSViolations = sum(violations);

    reward = normalizedD2DRate - lambdaQoS * numberOfQoSViolations;

end
