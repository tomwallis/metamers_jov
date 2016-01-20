% Function to run metamers experiment 10.
% Note that this uses functions and objects from the iShow toolbox, which
% must be present on matlab's search path for this script to work.

function metamers_experiment_10()
%% Set random seed:
seed = sum(100*clock);
reset(RandStream.getGlobalStream,seed);

%% Ask for some starting input:
param_ok = 0;
while param_ok == 0
    in = input('Are you in the lab? [y / n]', 's');
    % check parameters:
    if strcmp(in, 'y')
        lab = true;
        use_eyetracker = true;
        param_ok = 1;
    elseif strcmp(in, 'n')
        lab = false;
        use_eyetracker = false;
        param_ok = 1;
    else
        warning('Not a valid option. Try again.');
    end
end

if use_eyetracker ~= false
    param_ok = 0;
    while param_ok == 0
        in = input('Do you want to use the eyetracker? [y / n]', 's');
        % check parameters:
        if strcmp(in, 'y')
            use_eyetracker = true;
            param_ok = 1;
        elseif strcmp(in, 'n')
            use_eyetracker = false;
            param_ok = 1;
        else
            warning('Not a valid option. Try again.');
        end
    end
end

%% Parameters

% get parameters from files:
filepathToHere=pwd;
gen_file = fullfile(filepathToHere, '../stimuli/generation_params_exp_10.yaml');
gen_params = ReadYaml(gen_file);

% Trial durations in ms
ms_stim = 200;  % the duration each stimulus is shown to screen.
ms_isi = 500;  % the duration of blank screen between temporal intervals.
ms_resp = 1200;  % the duration for making a response (fixed interval ITI).
ms_fixation_check = 50;  % duration to check fixation.
ms_fixation = 500;  % duration of fixation at the start of a block.
ms_feedback = 100;  % duration of feedback.

ramp.alpha = 0.2;  % proportion of ms_stim that will be cosine ramping.

bg_color = 0.5;

dist_monitor = 60; % cm

ppd = gen_params.pix_per_deg;

% some fixation cross properties:
fix_colour_oval = [0 0 0];
fix_colour_normal = [0.7 0.7 0.7];
fix_colour_correct = [0.9 0.9 0.9];
fix_colour_wrong = [0.3 0.3 0.3];

% fixation cross location relative to monitor centre, in degrees:
fix_eccent_from_centre = 0;

% the angle of the patch from the fixation spot (in radians, ccw from
% right):
patch_angle_from_fixation = 0;

% patch position jitter range is a proportion of smallest patch size:
jitter = ppd;  % gen_params.patch_sizes{(1)} * 0.4;

% to make any motion artifact uninformative, jitter patch position along
% the tangent:
jitter_angle = jitter / gen_params.middle_centre_px; % max angular offset in radians...

% number of trials to do between breaks (and eyetracker validation, if
% requested).
break_trials = 50;

%% Start the hardware
if lab
    % load gamma correction file:
    calib = load('/home/data/calibration/lcd_gray/lcd_gray2015_04_01_1548.mat');

    clut = spline(calib.measurement, calib.input, linspace(0,1,(2^12))');
    clut(1:10) = 0;
    clut(clut<0)=0;

    % LCD-initialization
    win = window('lcd_gray', 'bg_color', bg_color, 'clut', clut);
    listener = listener_buttonbox('names', {'Green', 'White', 'Red'});
    waiter = listener_buttonbox('names', {'Green', 'White', 'Red'}, ...
            'does_interrupt', true);
    aud_volume = 0.5;
    aud = dpixx_audio_port('volume', aud_volume);
    aud.create_beep('short_low', 'low', .15, 0.25);
    aud.create_beep('short_high', 'high', .15, 0.25);
else
    %     win = window('debug', 'bg_color', bg_color, 'rect', [100, 100, 900, 900]);
    win = window('debug', 'bg_color', bg_color, 'rect', [0, 0, 800, 800]);
    listener = listener_keyboard('names', {'LeftArrow', 'DownArrow', 'RightArrow', 'Escape'});
    waiter = listener_keyboard('names', {'LeftArrow', 'DownArrow', 'RightArrow', 'Escape'}, ...
            'does_interrupt', true);
end


if use_eyetracker
    % initialise eyetracker
    try
        Eyelink('SetAddress','AN IP ADDRESS; REDACTED FOR SECURITY')

        % Provide Eyelink with details about the graphics environment
        % and perform some initializations. The information is returned
        % in a structure that also contains useful defaults
        % and control codes (e.g. tracker state bit and Eyelink key values).
        el=EyelinkInitDefaults(win.h);

        % Initialization of the connection with the Eyelink Gazetracker.
        Eyelink('Initialize','PsychEyelinkDispatchCallback');

        Eyelink('Initialize','PsychEyelinkDispatchCallback');

        [~, vs]=Eyelink('GetTrackerVersion');
        fprintf('Running experiment on a ''%s'' tracker.\n', vs );

        % make sure that we get gaze data from the Eyelink
        Eyelink('Command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA');

        eye_used = -1;

    catch eyelink_error
        % Shutdown Eyelink:
        Eyelink('Shutdown');
        % Close window:
        sca;
        % Restore keyboard output to Matlab:
        ListenChar(0);
        commandwindow;
        disp(eyelink_error);
        disp(eyelink_error.message);
        disp(eyelink_error.stack.line);
    end
end

%% Set file paths and open file to write to
data_path = fullfile(filepathToHere, '../../raw-data/experiment-10/');
im_path = fullfile(filepathToHere, '../../stimuli/experiment-10/final_ims/');
eye_data_path = fullfile(data_path, '/eye_data_files/');

if ~exist(data_path, 'dir')
  mkdir(data_path);
end

if ~exist(eye_data_path , 'dir')
  mkdir(eye_data_path);
end

%% Ask for parameters:

subj = input('Enter subject code:','s');

disp(['Check that viewing distance is ', num2str(dist_monitor), ' cm!!!'])

param_ok = 0;
while param_ok == 0
    patch_size = input('Which patch size to do? [32, 64, 128, 192, 256, 384, 512]:');
    % check parameters:
    if ~any([gen_params.patch_sizes{(1:end)}] == patch_size)
        warning('requested patch size is not a valid option. Try again.');
    else
        param_ok = 1;
    end
    [~, size_idx] = max([gen_params.patch_sizes{(1:end)}] == patch_size);
end

param_ok = 0;
while param_ok == 0
    surround = input('With surround ["y" or "n"]?:', 's');
    % check parameters:
    if strcmp(surround, 'y')
        % % present surround images.
        surround_cond = 'surround';
        param_ok = 1;
    elseif strcmp(surround, 'n')
        % present patches on blank background.
        surround_cond = 'blank';
        param_ok = 1;
    else
        warning('Not a valid option. Try again')
    end
end

%% Setup file saving structure
session = 1;

datafilename = fullfile(data_path, strcat('experiment-10_sub_',subj,'_session_', num2str(session), '.csv'));
% Check for existing result file to prevent accidentally overwriting files
% from a previous session:
while fopen(datafilename, 'rt')~=-1
    fclose('all');
    warning('File already exists. Using next session number.');
    session = session +1;
    datafilename = fullfile(data_path, strcat('experiment-10_sub_',subj,'_session_', num2str(session), '.csv'));
end

eyedata_fname = fullfile(eye_data_path, strcat('experiment-10_sub_',subj,'_session_', num2str(session), '.edf'));
% the remote file is stored on MS-DOS, so has filename length restrictions:
edf_file_remote = strcat('s-',subj,'-', num2str(session), '.edf');


%% Setup trial structure and counterbalancing

% load source image info table:
source_dat = readtable([im_path, 'patch_info.csv']);

% select source images corresponding to this size:
im_codes = table2array(source_dat(source_dat.size==patch_size, 'filename'));

% number of trials = num unique images, doubled for each condition of
% nat v synth, synth v synth:
[cond, im_code] = BalanceFactors(1, 0, {'nat_v_synth', 'synth_v_synth'}, im_codes);

odd_vec = repmat([0; 1], floor(length(cond)/4), 1);
while length(odd_vec) < floor(length(cond)/2)
    odd_vec = [odd_vec; randi(2, 1, 1)-1];
end
oddball = [Shuffle(odd_vec);
    Shuffle(odd_vec)];
% half the 'nat_v_synth' conditions are oddball natural.

% target position balanced for each condition:
target_vec = repmat([1; 2; 3], floor((length(im_code)/2) / 3), 1);
while length(target_vec) < floor(length(cond)/2)
    target_vec = [target_vec; randi(3, 1, 1)];
end
target_loc = [Shuffle(target_vec);
              Shuffle(target_vec)];
target_loc = num2str(target_loc);

% the surround conditions:
surround_cond = repmat(surround_cond, length(cond), 1);

% random determination of jitter on each trial:
jitter_1 = unifrnd(-jitter_angle, jitter_angle, length(im_code), 1);
jitter_2 = unifrnd(-jitter_angle, jitter_angle, length(im_code), 1);
jitter_3 = unifrnd(-jitter_angle, jitter_angle, length(im_code), 1);

data = table(cond, surround_cond, ...
             oddball, im_code, target_loc, ...
             jitter_1, jitter_2, jitter_3);

n_trials = height(data); % number of trials

% compute scale factor for these conditions:
scale = (patch_size / gen_params.pix_per_deg) / gen_params.middle_centre;

% fill new variables in data structure:
for trial=1:n_trials
    dat_struct(trial).subj = subj;
    dat_struct(trial).session = session;
    dat_struct(trial).eccent = gen_params.middle_centre;
    dat_struct(trial).patch_size_px = patch_size;
    dat_struct(trial).scale = scale;
    dat_struct(trial).test_location = 'sand';
    dat_struct(trial).rand_seed = seed;
    dat_struct(trial).trial = nan;
    dat_struct(trial).response = [];
    dat_struct(trial).rt = nan;
end

% add to table:
data = [data, struct2table(dat_struct)];

%% pseudo randomization-> ensure that an image never follows itself.

% for pilot testing:
if strcmp(subj, 'test')
    testmode = true;
    data = data(1:20, :);
    break_trials=10;
    n_trials = height(data); % number of trials
else
    testmode = false;
end

% Create random trial order
if testmode
    trial_row_idx = Shuffle(1:n_trials);
else
    trial_row_idx = pseudo_rand(data, 3);
end

%% Calculate spatial and temporal display workings.

% debug setup sometimes can't find framerate:
if lab == false && win.framerate == 0
    win.framerate = 60;
end

n_frames_stim = ceil(ms_stim * win.framerate / 1000);
n_frames_isi = ceil(ms_isi * win.framerate / 1000);
n_frames_resp = ceil(ms_resp * win.framerate / 1000);
n_frames_fixate = ceil(ms_fixation * win.framerate / 1000);
n_frames_feedback = ceil(ms_feedback * win.framerate / 1000);

ramp.size = n_frames_stim;
temporal_ramp = curve_tapered_cosine(ramp);  % function from ishow. 150 ms at full contrast.

fix_position = [(win.rect(3) / 2) + fix_eccent_from_centre * ppd, win.rect(4) / 2];

experiment_start_time = tic;

%% Calibrate eyetracker, if used.
if use_eyetracker
    try
        % open file to record data to
        Eyelink('Openfile', edf_file_remote);

        % Calibrate the eye tracker
        EyelinkDoTrackerSetup(el);

        %         % do a final check of calibration using driftcorrection
        %         EyelinkDoDriftCorrection(el);

    catch eyelink_error
        % Shutdown Eyelink:
        Eyelink('Shutdown');
        % Close window:
        sca;
        % Restore keyboard output to Matlab:
        ListenChar(0);
        commandwindow;
        disp(eyelink_error);
        disp(eyelink_error.message);
        disp(eyelink_error.stack.line);
    end
end

%% Trials
try
    ListenChar(2);

    for trial=1:n_trials

        this_trial_idx = trial_row_idx(trial);  % the index of this trial in the data frame.

        % Save properties in data structure
        data.trial(this_trial_idx) = trial;

        if trial == 1
            win.pause_trial(waiter, 'Press any button to start!');
            for itic = 1 : n_frames_fixate
                win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
                win.flip();
            end
        end

        %% Load parameters for this trial
        im_code = table2array(data(this_trial_idx, 'im_code'));
        cond = table2array(data(this_trial_idx, 'cond'));
        patch_size = table2array(data(this_trial_idx, 'patch_size_px'));
        surround_cond = table2array(data(this_trial_idx, 'surround_cond'));
        oddball = table2array(data(this_trial_idx, 'oddball'));
        target_loc = table2array(data(this_trial_idx, 'target_loc'));
        jitter_1 = table2array(data(this_trial_idx, 'jitter_1'));
        jitter_2 = table2array(data(this_trial_idx, 'jitter_2'));
        jitter_3 = table2array(data(this_trial_idx, 'jitter_3'));

        if iscell(im_code)
            im_code = im_code{(1)};
        end

        %% target images:

        if strcmp(cond, 'nat_v_synth')
            % natural vs synth trial
            im_0_fname = get_image_fname(im_code);  % the natural image
            im_0_fname = char(fullfile(im_path, im_0_fname));
            im_1_fname = get_synth_fname(im_code, 1);  % synthetic image 1
            im_1_fname = char(fullfile(im_path, im_1_fname));
        elseif strcmp(cond, 'synth_v_synth')
            % synth vs synth trial
            im_0_fname = get_synth_fname(im_code, 2);  % synth 2
            im_0_fname = char(fullfile(im_path, im_0_fname));
            im_1_fname = get_synth_fname(im_code, 3);  % synth 3
            im_1_fname = char(fullfile(im_path, im_1_fname));
        else
            disp('Couldn''t match experimental condition!');
        end

        % load images, make textures. Matlab needs to read alpha channel
        % separately for some reason...
        [im_0_tex, ~] = make_image_texture(im_0_fname);
        [im_1_tex, ~] = make_image_texture(im_1_fname);

        % determine target / nontarget texture according to oddball:
        if oddball == 0
            targ_tex = im_0_tex;
            nontarg_tex = im_1_tex;
        elseif oddball == 1
            targ_tex = im_1_tex;
            nontarg_tex = im_0_tex;
        end

        % determine interval texture according to target_pos:
        if strcmp(target_loc, '1')
            tex_1 = targ_tex;
            tex_2 = nontarg_tex;
            tex_3 = nontarg_tex;
        elseif strcmp(target_loc, '2')
            tex_1 = nontarg_tex;
            tex_2 = targ_tex;
            tex_3 = nontarg_tex;
        elseif strcmp(target_loc, '3')
            tex_1 = nontarg_tex;
            tex_2 = nontarg_tex;
            tex_3 = targ_tex;
        end

        % determine drawing rects:
        middle_radius = gen_params.middle_centre_px;
        middle_rect_base = [0, 0, patch_size, patch_size];

        % position in each interval:
        middle_1 = jittered_rect(fix_position, middle_radius, patch_angle_from_fixation + jitter_1, middle_rect_base);
        middle_2 = jittered_rect(fix_position, middle_radius, patch_angle_from_fixation + jitter_2, middle_rect_base);
        middle_3 = jittered_rect(fix_position, middle_radius, patch_angle_from_fixation + jitter_3, middle_rect_base);

        % texture rotation angles (to align contours despite jitter):
        rotate_1 = (patch_angle_from_fixation + jitter_1) * (180 / pi);
        rotate_2 = (patch_angle_from_fixation + jitter_2) * (180 / pi);
        rotate_3 = (patch_angle_from_fixation + jitter_3) * (180 / pi);

        %% surrounds:

        if strcmp(surround_cond, 'surround')
            inner_fname = get_synth_fname(im_code, 4);  % synth 4
            inner_fname = char(fullfile(im_path, inner_fname));
            outer_fname = get_synth_fname(im_code, 5);  % synth 5
            outer_fname = char(fullfile(im_path, outer_fname));

            % load images, make textures. Matlab needs to read alpha channel
            % separately for some reason...
            [inner_tex, ~] = make_image_texture(inner_fname);
            [outer_tex, ~] = make_image_texture(outer_fname);

            inner_radius = middle_radius - patch_size;
            outer_radius = middle_radius + patch_size;

            % % determine drawing rects:
            inner_rect_base = [0, 0, patch_size, patch_size];
            outer_rect_base = [0, 0, patch_size, patch_size];

            % determine patch jitter in each interval:
            inner_1 = jittered_rect(fix_position, inner_radius, patch_angle_from_fixation + jitter_1, inner_rect_base);
            outer_1 = jittered_rect(fix_position, outer_radius, patch_angle_from_fixation + jitter_1, outer_rect_base);

            inner_2 = jittered_rect(fix_position, inner_radius, patch_angle_from_fixation + jitter_2, inner_rect_base);
            outer_2 = jittered_rect(fix_position, outer_radius, patch_angle_from_fixation + jitter_2, outer_rect_base);

            inner_3 = jittered_rect(fix_position, inner_radius, patch_angle_from_fixation + jitter_3, inner_rect_base);
            outer_3 = jittered_rect(fix_position, outer_radius, patch_angle_from_fixation + jitter_3, outer_rect_base);
        end



        %% start eye recording, check that fixation is within 2 degrees of middle.
        % wait ITI.
        if use_eyetracker
            % start recording eye position
            Eyelink('StartRecording');
            WaitSecs(0.01);
            Eyelink('StartRecording');

            %%%%%% check initial fixation is within 2deg of the spot %%%%%
            % code here adapted from Will Harrison...

            checkFix = 0;

            while checkFix == 0

                % check for presence of a new sample update
                if Eyelink('NewFloatSampleAvailable') > 0

                    evt = Eyelink('NewestFloatSample');

                    if eye_used ~= -1 % do we know which eye to use yet?
                        % if we do, get current gaze position from sample
                        x = evt.gx(eye_used+1); % +1 as we're accessing MATLAB array
                        y = evt.gy(eye_used+1);

                        fixCheckX = abs(x-(fix_position(1)));
                        fixCheckY = abs(y-fix_position(2));

                        % do we have valid data and is the pupil visible?
                        if x~=el.MISSING_DATA && y~=el.MISSING_DATA && evt.pa(eye_used+1)>0

                            fixedTimer = GetSecs;
                            tic;
                            while ((fixCheckX < ppd*2) && (fixCheckY < ppd*2));

                                win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
                                win.flip();

                                evt = Eyelink('NewestFloatSample');

                                x = evt.gx(eye_used+1);
                                y = evt.gy(eye_used+1);

                                fixCheckX = abs(x-(fix_position(1)));
                                fixCheckY = abs(y-fix_position(2));
                                fixTime = GetSecs - fixedTimer;

                                if fixTime > (ms_fixation_check / 1000)

                                    checkFix = 1;
                                    break;

                                end

                            end;

                            checkFixTimer = GetSecs;

                            if checkFix < 1

                                while ((fixCheckX > ppd*2) || (fixCheckY > ppd*2));

                                    win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
                                    win.flip();

                                    evt = Eyelink('NewestFloatSample');
                                    x = evt.gx(eye_used+1);
                                    y = evt.gy(eye_used+1);
                                    fixCheckX = abs(x-(fix_position(1)));

                                    fixCheckY = abs(y-fix_position(2));

                                    totalCheckFixTime = GetSecs - checkFixTimer;

                                    if totalCheckFixTime > 2

                                        win.draw_text(['The tracker thinks you are '...
                                            'not looking at the fixation spot. \n\n Recalibrating...']);
                                        win.flip();
                                        WaitSecs(1);

                                        EyelinkDoTrackerSetup(el);

                                        % start recording eye position again
                                        Eyelink('StartRecording');
                                        tic;
                                        checkFixTimer = GetSecs;

                                    end;

                                end

                            end

                        end
                    else % if we don't, first find eye that's being tracked
                        if 0
                            eye_used = el.RIGHT_EYE;
                        else
                            eye_used = Eyelink('EyeAvailable'); % get eye that's tracked
                            if eye_used == el.BINOCULAR; % if both eyes are tracked
                                eye_used = el.LEFT_EYE; % use left eye
                            end
                        end
                    end
                end
            end;

            %%%%%% check initial fixation is within 2deg of the spot %%%%%
        else

        end

        if lab
            aud.play('short_high');
            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            win.flip();
        end

        % mark zero-plot time in data file
        if use_eyetracker
            Eyelink('Message', ['start_trial: ' num2str(trial)]);
        end

        %% stimulus - gap intervals.

        % interval 1:
        for itic = 1 : n_frames_stim
            win.draw(tex_1, temporal_ramp(itic), middle_1, rotate_1);
            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            if strcmp(surround_cond, 'surround')
                win.draw(inner_tex, temporal_ramp(itic), inner_1, rotate_1);
                win.draw(outer_tex, temporal_ramp(itic), outer_1, rotate_1);
            end
            win.flip();
        end

        if use_eyetracker
            Eyelink('Message', ['end_interval: ' num2str(1)]);
        end

        % isi:
        for itic = 1 : n_frames_isi
            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            win.flip();
        end

        % interval 2:
        for itic = 1 : n_frames_stim
            win.draw(tex_2, temporal_ramp(itic), middle_2, rotate_2);
            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            if strcmp(surround_cond, 'surround')
                win.draw(inner_tex, temporal_ramp(itic), inner_2, rotate_2);
                win.draw(outer_tex, temporal_ramp(itic), outer_2, rotate_2);
            end
            win.flip();
        end

        if use_eyetracker
            Eyelink('Message', ['end_interval: ' num2str(2)]);
        end

        % isi:
        for itic = 1 : n_frames_isi
            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            win.flip();
        end

        % interval 3:
        for itic = 1 : n_frames_stim
            win.draw(tex_3, temporal_ramp(itic), middle_3, rotate_3);
            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            if strcmp(surround_cond, 'surround')
                win.draw(inner_tex, temporal_ramp(itic), inner_3, rotate_3);
                win.draw(outer_tex, temporal_ramp(itic), outer_3, rotate_3);
            end
            win.flip();
        end

        if use_eyetracker
            Eyelink('Message', ['end_interval: ' num2str(3)]);
        end

        % clear the screen:
        win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
        win.flip();

        % flag trial end:
        if use_eyetracker
            Eyelink('Message', 'end_trial');
        end

        if lab  % tone to signal trial end.
            aud.play('short_high');
            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            win.flip();
        end

        %% Wait for a response, or response interval to be exceeded.
%         listener.start();
%         caught_response = false;
%         try
%             for itic = 1 : n_frames_resp
%                 %while caught_response == false
%                 win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
%                 win.flip();
%                 listener.check();
%             end
%         catch e
%             % If the error-identifier does not match the expected id
%             % ?iShow:ResponseInterrupt?, we should rethrow the error, otherwise
%             % errors could be occuring in our program without us noticing.
%             if strcmp(e.identifier, 'iShow:ResponseInterrupt')
%                 caught_response = true;
%             else
%                 rethrow(e);
%             end
%         end
%
%         % If we registered a response, the listener has stopped automatically,
%         % however if we haven?t, we have to do this manually.
%         if ~caught_response
%             listener.stop();
%         end
%
%         [press, rt] = listener.response.get_presses('first');
%
%         %%%%%%% button press continues to next trial %%%%%%


        %%%%%% fixed timing response interval %%%%%%
        % Response interval
        listener.start();
        for itic = 1 : n_frames_resp
            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            win.flip();
        end
        response = listener.stop();

        % Save the responses
        [press, rt] = listener.response.get_presses('first');

        %%%%% fixed timing response interval %%%%%%

        switch press
            case 1
                res = '1';
            case 2
                res = '2';
            case 3
                res = '3';
            otherwise
                res = 'na';
        end

        data.response(this_trial_idx) = {res};
        data.rt(this_trial_idx) = rt;

        if use_eyetracker
            Eyelink('Message', 'end_response');
            Eyelink('Message', ['response = ', res]);
        end

        % close unused textures to save memory:
        Screen('Close', [im_0_tex, im_1_tex]);
        if strcmp(surround_cond, 'surround')
            Screen('Close', [inner_tex, outer_tex]);
        end

        % provide feedback via fixation cross colour:
        if strcmp(res, target_loc)
            for itic = 1 : n_frames_feedback
                win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_correct);
                win.flip();
            end
        else
            if lab
                aud.play('short_low');
                win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
                win.flip();
            end
            for itic = 1 : n_frames_feedback
                win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_wrong);
                win.flip();
            end
        end

        if use_eyetracker
            Eyelink('Message', 'end_feedback');
            Eyelink('StopRecording');
            WaitSecs(0.01);
            Eyelink('StopRecording');
        end

        % if this is a trial break:
        if mod(trial,break_trials)==0 && trial~=n_trials
            trial_sum = 0;
            for icorr=(trial-(break_trials-1)):trial
                this_trial_idx = trial_row_idx(icorr);
                if strcmp(data.response(this_trial_idx), data.target_loc(this_trial_idx))
                    trial_sum = trial_sum+1;
                end
            end
            perc_corr = (trial_sum/break_trials)*100;

            win.pause_trial(waiter, ...
                sprintf(['%d out of the last %d trials correct. '...
                'That corresponds to %.2f %% correct. \n'...
                'You have finished %d blocks out of %d. \n'...
                '\nPress a key to continue!'], ...
                trial_sum,break_trials,perc_corr,...
                (trial/break_trials), round(n_trials/break_trials)));

            win.draw_fixation(ppd, fix_position, fix_colour_oval, fix_colour_normal);
            win.flip();
            WaitSecs(ms_fixation/1000);
        end

        % Responded to last trial
        if trial == n_trials
            trial_sum = 0;
            for icorr=(trial-(n_trials-1)):trial
                if strcmp(data.response(icorr), data.target_loc(icorr))
                    trial_sum = trial_sum+1;
                end
            end
            perc_corr = (trial_sum/n_trials)*100;

            win.pause_trial(waiter, ...
                sprintf(['%d out of %d trials correct. '...
                'That corresponds to %.2f %% correct. \n' ...
                'Press a key to finish!'], ...
                trial_sum,n_trials,perc_corr));
        end
    end

    if lab == false
        Screen('CloseAll');
    end

    ListenChar(1);

catch e
    ListenChar(1);
    fclose('all');
    Screen('CloseAll');
    rethrow(e);
    ListenChar(1);

    if use_eyetracker
        % Shutdown Eyelink:
        Eyelink('Shutdown');
    end
end

%% Write results to files using Matlab's amazing new Tables (WELCOME TO THE FUTURE TM):

% sort by trial:
data = sortrows(data, 'trial');
writetable(data, datafilename);

if use_eyetracker
    Eyelink('Command', 'clear_screen 0')
    Eyelink('CloseFile');
    % download data file
    try
        fprintf('Receiving data file ''%s''\n', edf_file_remote );
        status=Eyelink('ReceiveFile',[],eye_data_path,1);
        pause(1);

        movefile([eye_data_path, edf_file_remote], eyedata_fname);
        if status > 0
            fprintf('ReceiveFile status %d\n', status);
        end
        if 2==exist(edf_file_remote, 'file')
            fprintf('Data file ''%s'' can be found in ''%s''\n', edf_file_remote, eye_data_path );
        end
    catch
        fprintf('Problem receiving data file ''%s''\n', edf_file_remote );
    end
    % Shutdown Eyelink:
    Eyelink('Shutdown');
end

fclose('all');

experiment_duration = toc(experiment_start_time) / 60;
disp(sprintf('The experiment took %2.1f minutes', experiment_duration))

%% Some subfunctions

    function trial_order=pseudo_rand(data, max_consec)
        % this function adapted from Heiko's `short_trials.m`.
        % max_consec = How many trials between same image allowed >=1 !
        n_t = height(data); % number of trials

        idx = Shuffle(1:n_t);

        accepted = false;

        while ~accepted
            im_vec = data.im_code;
            % FUCK matlab cell arrays...
            im_vec = im_vec(idx);
            for i = 1 : (n_t - max_consec)
                this_im = im_vec(i);

                for j = 1 : max_consec
                    % if the next trial is the same image, jump to a random spot.
                    if  strcmp(im_vec(i + j), this_im)
                        who_jumps = i + j;
                        where_to_jump = randi(n_t);
                        % swap indices:
                        displaced_idx = idx(where_to_jump);
                        idx(where_to_jump) = idx(who_jumps);
                        idx(who_jumps) = displaced_idx;  % swapped with random int.
                    end
                end
            end

            % do a check loop:
            problem = false;
            for i = 1 : (n_t - max_consec)
                this_im = im_vec(i);
                if any(strcmp(im_vec((i+1):(i+max_consec)), this_im))
                    problem = true;
                end
            end

            if ~problem
                accepted = true;
                disp('success in pseudorandom trial ordering');
            else
                disp('trial ordering failed, looping again');
            end
        end
        trial_order = idx;
    end


    function fname = get_image_fname(im_code)
        fname = sprintf('%s_natural.png',im_code);
    end


    function fname = get_synth_fname(im_code, synth_num)
        fname = sprintf('%s_synth_%1d.png',im_code, synth_num);
    end


    function rect = jittered_rect(fix_position, radius, angle, rect_base)
        centre = compute_patch_centre(fix_position, radius, angle);
        rect = CenterRectOnPoint(rect_base, centre(1), centre(2));
    end


    function patch_centre = compute_patch_centre(fix_position, radius, angle)
        % fix position in screen coords, radius in pixels, angle in radians
        % (ccw from right).
        patch_centre = [(cos(angle) * radius) + fix_position(1), ...
            (sin(angle) * radius) + fix_position(2)];
    end

    function [tex, im] = make_image_texture(fname)
        [im, ~, alpha] = imread(fname);  % image is an RGBA image.
        im(:, :, 4) = alpha;
        im = im2double(im);
        tex = win.make_texture(im);
    end

end
