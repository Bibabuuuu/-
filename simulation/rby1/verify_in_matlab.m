function report = verify_in_matlab()
% Run in MATLAB with Robotics System Toolbox. Not yet run in this workspace.
% This checks fixed-base link dynamics, not wheel-ground/contact dynamics.
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
output = fullfile(root, 'outputs', 'rby1-research');
robot = importrobot(fullfile(output, 'model', 'nominal_fixed_base.urdf'), ...
    'DataFormat', 'struct');
home = homeConfiguration(robot);
jointNames = {home.JointName};
robot.DataFormat = 'column';
robot.Gravity = [0 0 -9.81];
fixtures = jsondecode(fileread(fullfile(output, 'matlab_fixtures.json')));
errors = zeros(numel(fixtures), 3);
for k = 1:numel(fixtures)
    sample = fixtures(k);
    [found, map] = ismember(jointNames, sample.joint_order);
    assert(all(found), 'MATLAB joint order does not map to the source model.');
    q = zeros(numel(jointNames), 1);
    for j = 1:numel(jointNames)
        q(j) = sample.q_by_joint.(jointNames{j});
    end
    actualM = massMatrix(robot, q);
    expectedM = sample.fixed_base_M_without_armature(map, map);
    actualG = gravityTorque(robot, q);
    expectedG = sample.gravity_compensation(map);
    errors(k, 1) = max(abs(actualM - expectedM), [], 'all');
    errors(k, 2) = max(abs(actualG(:) - expectedG(:)));
    fkError = 0;
    for name = {'ee_left', 'ee_right'}
        T = getTransform(robot, q, name{1});
        expected = sample.ee_positions_in_base.(name{1});
        fkError = max(fkError, max(abs(T(1:3, 4) - expected(:))));
    end
    errors(k, 3) = fkError;
    assert(all(abs(actualM-expectedM) <= 1e-7 + 1e-5*abs(expectedM), 'all'));
    assert(all(abs(actualG(:)-expectedG(:)) <= 1e-7 + 1e-5*abs(expectedG(:))));
    assert(fkError < 1e-6);
end
report = struct('passed', true, 'matlab_version', version, ...
    'columns', {{'mass_matrix_error', 'gravity_error', 'position_error_m'}}, ...
    'errors', errors, 'scope', 'Fixed-base rigid dynamics, rotor armature excluded');
fid = fopen(fullfile(output, 'matlab_validation.json'), 'w');
assert(fid ~= -1);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(report, 'PrettyPrint', true));
disp(report);
end
