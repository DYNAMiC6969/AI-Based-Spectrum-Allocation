%% Stage 7 - Channel Gain / Path-Loss (D2D-to-D2D only)
% AI-Based Dynamic Spectrum Allocation Using Q-Learning in D2D-Enabled
% Cellular Networks
%
% SCOPE OF THIS STAGE:
% This script computes the distance-dependent channel gain between every
% D2D transmitter and every D2D receiver, using the fixed positions from
% Stage 6. It does NOT implement interference, SINR, reward, or
% Q-learning — those are later stages.
%
% NOTE: beta and alpha values below are OUR simulation assumptions for
% this test, not values taken from the professor-provided thesis.

clear; clc; close all;

%% 1. Load fixed network positions (Stage 6)
net = generatePositions();

%% 2. Path-loss model parameters (our simulation assumptions)
beta  = 1;    % path-loss constant (assumption, dimensionless for this test)
alpha = 3;    % path-loss exponent (assumption, typical range 2-4)

%% 3. Calculate D2D transmitter-to-receiver distances
numPairs = size(net.D2D_TX, 1);   % = 2

D = zeros(numPairs, numPairs);    % D(i,j) = distance: D2D_TX_i -> D2D_RX_j
for i = 1:numPairs
    for j = 1:numPairs
        D(i,j) = calculateDistance(net.D2D_TX(i,:), net.D2D_RX(j,:));
    end
end

disp('--- D2D TX-to-RX Distance Matrix (m) ---');
disp(D);

%% 4. Calculate the D2D channel-gain matrix H
% H(i,j) = channel gain from D2D_TX_i to D2D_RX_j
% Diagonal entries (H11, H22) = desired D2D links
% Off-diagonal entries (H12, H21) = cross/interference links
H = channelGain(D, beta, alpha);

disp(' ');
disp('--- D2D Channel Gain Matrix H ---');
disp(H);

fprintf('\nDesired link gains:  h11 = %.6g,  h22 = %.6g\n', H(1,1), H(2,2));
fprintf('Cross-link gains:    h12 = %.6g,  h21 = %.6g\n', H(1,2), H(2,1));

%% 5. Verification Test 1: Desired links have larger gain than cross links
% (Desired links use the short intra-pair distance; cross links use the
%  much longer inter-pair distance, so this should hold given Stage 6's
%  geometry.)
desiredGains = [H(1,1), H(2,2)];
crossGains   = [H(1,2), H(2,1)];

test1_pass = all(min(desiredGains) > max(crossGains));
fprintf('\nTest 1 (desired gain > cross gain): %s\n', mat2str(test1_pass));

%% 6. Verification Test 2: Smaller distance => larger gain (monotonicity)
r_near = 5;    % meters
r_far  = 50;   % meters
h_near = channelGain(r_near, beta, alpha);
h_far  = channelGain(r_far, beta, alpha);

test2_pass = h_near > h_far;
fprintf('Test 2 (smaller distance -> larger gain): %s  (h(%dm)=%.6g, h(%dm)=%.6g)\n', ...
    mat2str(test2_pass), r_near, h_near, r_far, h_far);

%% 7. Verification Test 3: Larger alpha => stronger attenuation
alpha_low  = 2;
alpha_high = 4;
r_test     = 20;  % meters

h_alpha_low  = channelGain(r_test, beta, alpha_low);
h_alpha_high = channelGain(r_test, beta, alpha_high);

test3_pass = h_alpha_high < h_alpha_low;
fprintf('Test 3 (larger alpha -> stronger attenuation): %s  (alpha=%d: h=%.6g, alpha=%d: h=%.6g)\n', ...
    mat2str(test3_pass), alpha_low, h_alpha_low, alpha_high, h_alpha_high);

%% 8. Verification Test 4: No NaN/Inf values in H
test4_pass = all(isfinite(H(:)));
fprintf('Test 4 (no NaN/Inf in channel-gain matrix): %s\n', mat2str(test4_pass));

%% 9. Overall summary
disp(' ');
if test1_pass && test2_pass && test3_pass && test4_pass
    disp('All Stage 7 verification tests PASSED.');
else
    disp('One or more Stage 7 verification tests FAILED — check model/parameters.');
end
