% Matlab script to run texture synthesization on patches.
%
% tsawallis wrote it.

% add the pyramid and texture toolboxes to matlab's search path:
addpath(genpath('/home/tomw/matlab_toolboxes/matlabPyrTools'))
addpath(genpath('/home/tomw/matlab_toolboxes/matlabPyrTools/MEX'))
addpath(genpath('/home/tomw/matlab_toolboxes/textureSynth'))
addpath(genpath('/home/tomw/matlab_toolboxes/textureSynth/MEX'))

s = RandStream('mt19937ar', 'Seed', 1346537);
RandStream.setGlobalStream(s);

% paths to images:

% (because matlab sucks at strings and directories, this code sucks:
this_dir = pwd;
top_dir = this_dir(1:end-12);
raw_path = [top_dir, 'stimuli/experiment-9/inner_patches'];
out_path = [top_dir, 'stimuli/experiment-9/inner_synths'];

if ~exist(out_path, 'dir')
  mkdir(out_path);
end

% ims:
ims = dir([raw_path, '/*.png']);


%% Parameters

n_unique = 1; % the number of unique textures to save for each original patch.

Nor = 4; % Number of orientations
Na = 9;  % Spatial neighborhood is Na x Na coefficients
% It must be an odd number!

Niter = 50;	% Number of iterations of synthesis loop

n_par_cores = 12;  % the number of compute cores available.

%% normal for loop

% for i = 1 : length(ims)
% for i = 1 : 1
%     file = ims(i).name;
%     im = imread([raw_path, '/', file]);
%     im = double(im);
%
%     im_size = size(im, 1);
%     Nsc = log2(im_size) - 2; % use max number of scales for this image size.
%     Nsc = min(Nsc, 5);  % use max 5 scales.
%
%     Nsx = size(im, 1);	% Size of synthetic image is Nsy x Nsx
%     Nsy = size(im, 1);	% WARNING: Both dimensions must be multiple of 2^(Nsc+2)
%
%     params = textureAnalysis(im, Nsc, Nor, Na);
%
%     for j = 1 : n_unique
%         success = 0;
%         while success == 0
%             try
%                 res = textureSynthesis(params, [Nsy Nsx], Niter);
%
%                 fname = [out_path, '/', file(1:end-4), ...
%                     '_synth_', num2str(j), '.png'];
%                 res = uint8(res);
%                 imwrite(res, fname);
%
%                 success = 1;
%
%             catch res_failed
%
%             end
%         end
%     end
% end

%% Parallel for loop, if required:

try
    % open parallel pool:
    pool = parpool(n_par_cores);
% %     matlabpool(8); % when running on lovelace (matlab2010a).

    parfor i = 1 : length(ims)
        file = ims(i).name;
        im = imread([raw_path, '/', file]);
        im = double(im);

        im_size = size(im, 1);
        Nsc = log2(im_size) - 2; % max number of scales for this image size.
        Nsc = min(Nsc, 4);  % use max 4 scales.

        Nsx = size(im, 1);	% Size of synthetic image is Nsy x Nsx
        Nsy = size(im, 1);	% WARNING: Both dimensions must be multiple of 2^(Nsc+2)

        params = textureAnalysis(im, Nsc, Nor, Na);

        for j = 1 : n_unique
            success = 0;
            attempt_count = 0;
            while success == 0  && attempt_count <= 10
                try
                    disp(['Synthesising stim ' file])
                    res = textureSynthesis(params, [Nsy Nsx], Niter);

                    fname = [out_path, '/', file(1:end-4), ...
                        '_synth_', num2str(j), '.png'];
                    res = uint8(res);
                    imwrite(res, fname);

                    success = 1;

                catch res_failed
                    attempt_count = attempt_count + 1;
                    if attempt_count == 11
                        disp(['Cant do image ' file '; moving on!'])
                    else
                        disp('Synth failed, trying again!')
                    end

                end
            end
        end
    end

    delete(gcp('nocreate'))
%     matlabpool close
catch ME
    delete(gcp('nocreate'))
%     matlabpool close

end


