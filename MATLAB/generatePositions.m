function net = generatePositions()
% GENERATEPOSITIONS  Returns fixed initial node positions for Stage 6.
%
% NOTE: These coordinates are OUR simulation assumptions for this
% initial test — they are not taken from the professor-provided thesis.
%
% Output:
%   net.BS       - [1x2] Base Station coordinate
%   net.CU       - [3x2] Cellular User coordinates
%   net.D2D_TX   - [2x2] D2D transmitter coordinates
%   net.D2D_RX   - [2x2] D2D receiver coordinates
%   net.simArea  - [xmin xmax ymin ymax] simulation area bounds (meters)

    net.BS = [50 50];

    net.CU = [20 80;
              80 60;
              70 20];

    net.D2D_TX = [25 35;
                  75 80];

    net.D2D_RX = [30 38;
                  78 75];

    net.simArea = [0 100 0 100];  % 100 m x 100 m simulation area

end
