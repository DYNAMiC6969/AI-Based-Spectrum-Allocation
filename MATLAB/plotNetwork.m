function plotNetwork(net)
% PLOTNETWORK  Visualizes the Base Station, Cellular Users, D2D pairs,
% and D2D links within the simulation area.
%
% Input:
%   net - struct returned by generatePositions()

    figure;
    hold on;

    % Base Station
    plot(net.BS(1), net.BS(2), 'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'g');

    % Cellular Users
    plot(net.CU(:,1), net.CU(:,2), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

    % D2D Transmitters
    plot(net.D2D_TX(:,1), net.D2D_TX(:,2), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');

    % D2D Receivers
    plot(net.D2D_RX(:,1), net.D2D_RX(:,2), 'co', 'MarkerSize', 8, 'MarkerFaceColor', 'c');

    % D2D links (TX -> RX)
    numPairs = size(net.D2D_TX, 1);
    for j = 1:numPairs
        plot([net.D2D_TX(j,1) net.D2D_RX(j,1)], ...
             [net.D2D_TX(j,2) net.D2D_RX(j,2)], 'k--', 'LineWidth', 1.5);
    end

    xlim(net.simArea(1:2));
    ylim(net.simArea(3:4));
    xlabel('X-axis (m)');
    ylabel('Y-axis (m)');
    title('Stage 6: Network Topology (BS, CUs, D2D Pairs)');
    legend('Base Station', 'Cellular Users', 'D2D Transmitters', 'D2D Receivers', 'D2D Links', ...
           'Location', 'bestoutside');
    grid on;
    axis square;
    hold off;

end
