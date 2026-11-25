%% Stage 6 - Network Generation and Visualization
% AI-Based Dynamic Spectrum Allocation Using Q-Learning in D2D-Enabled
% Cellular Networks
%
% SCOPE OF THIS STAGE:
% This script ONLY generates and visualizes the physical network
% topology and verifies node distances/placement. Q-learning, the
% Q-table, reward, SINR, and interference are NOT implemented here —
% those are introduced in later stages.

clear; clc; close all;

%% 1. Generate node positions
net = generatePositions();

%% 2. Plot the network
plotNetwork(net);

%% 3. Calculate D2D link distances
numD2DPairs = size(net.D2D_TX, 1);
d2dDistances = zeros(numD2DPairs, 1);

for j = 1:numD2DPairs
    d2dDistances(j) = calculateDistance(net.D2D_TX(j,:), net.D2D_RX(j,:));
end

disp('--- D2D Pair Link Distances (m) ---');
for j = 1:numD2DPairs
    fprintf('D2D Pair %d: TX(%.1f, %.1f) -> RX(%.1f, %.1f)   Distance = %.4f m\n', ...
        j, net.D2D_TX(j,1), net.D2D_TX(j,2), net.D2D_RX(j,1), net.D2D_RX(j,2), d2dDistances(j));
end

%% 4. Calculate BS-to-CU distances
numCUs = size(net.CU, 1);
cuDistances = zeros(numCUs, 1);

for i = 1:numCUs
    cuDistances(i) = calculateDistance(net.BS, net.CU(i,:));
end

disp(' ');
disp('--- Base Station to Cellular User Distances (m) ---');
for i = 1:numCUs
    fprintf('CU %d: (%.1f, %.1f)   Distance to BS = %.4f m\n', ...
        i, net.CU(i,1), net.CU(i,2), cuDistances(i));
end

%% 5. Verify all nodes lie within the simulation area
allPositions = [net.BS; net.CU; net.D2D_TX; net.D2D_RX];
xMin = net.simArea(1); xMax = net.simArea(2);
yMin = net.simArea(3); yMax = net.simArea(4);

withinBounds = all(allPositions(:,1) >= xMin & allPositions(:,1) <= xMax & ...
                    allPositions(:,2) >= yMin & allPositions(:,2) <= yMax);

disp(' ');
if withinBounds
    disp('Topology check PASSED: all nodes lie within the 100 m x 100 m simulation area.');
else
    disp('Topology check FAILED: one or more nodes lie outside the simulation area.');
end
