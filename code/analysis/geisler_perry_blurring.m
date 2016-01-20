function geisler_perry_blurring
%% function that outputs images from Experiment 1
% that are blurred by the svis toolbox.

this_dir = pwd;
top_dir = this_dir(1:end-14);
in_path = fullfile(top_dir, 'stimuli', 'experiment-13', 'final_ims');
out_path = fullfile(top_dir, 'code', 'analysis', 'geisler_perry_ims');

% params:
halfres = 2.3;  % set to default for humans (2.3)
ppd = 43;  % pixels per degree from our experiment.
fix_pos = [256, 12];  % position of fovea (row, col)

% % demo im:
% ims = dir([in_path, '/i2237929211']);
% this_im = ims(1).name;

% read filenames:
im_dat = readtable(fullfile(out_path, 'im_data.csv')); % written out by spectral analysis ipynb.

% Initialize the library
svisinit

for i = 1 : height(im_dat)
    im_id = table2array(im_dat(i, 'filename'));
    im_id = im_id{1};  % to string, because matlab.
    
    % load:
    [img, rows, cols] = load_image(im_id);
    source_size = [rows, cols];
    
    % background:
    [img, rows, cols] = background_image(img, fix_pos);
    
    % Create a resmap
    maparc = 2 * cols/ppd; % half of this is the image's horizontal size in degrees.
    % fprintf('Creating resolution map...\n');
    resmap=svisresmap(rows*2,cols*2,'halfres',halfres,'maparc',maparc);
    
    do_blur(img, resmap, fix_pos, source_size);
    
end

% Free resources
svisrelease


%% Helper functions from here
    function [img, rows, cols] = load_image(filename)
        % Read in the file
        fprintf('Reading %s...\n',filename);
        fname = sprintf('%s_mid_nat.png', fullfile(in_path, filename));
        [img, ~, ~] = imread(fname);
        img = rgb2gray(img);
        rows=size(img,1);
        cols=size(img,2);
    end

    function [img, rows, cols] = background_image(img, fixation)
        % create a mean grey background image:
        bg = ones(512, 768);
        bg(:) = bg(:) * 128;
        bg = uint8(bg);
        
        % create a "fixation spot":
        width = 3;
        bg(fixation(1) - width:fixation(1) + width, fixation(2)-width: fixation(2)+width) = 255;
        
        % alpha blending seems complicated; I don't have functions to do it in matlab. Just graft on:
        im_loc = fixation;
        im_loc(2) = im_loc(2) + 430;  % distance of centre from fovea in px.
        
        bg(im_loc(1) - size(img,1)/2: im_loc(1) + size(img,1)/2-1,...
            im_loc(2) - size(img,2)/2: im_loc(2) + size(img,2)/2-1) = img;
        
        img = bg;
        
        rows=size(img,1);
        cols=size(img,2);
    end


%     function res = my_resmap()
%         % Create a resolution map from the Geisler and Perry (1998)
%         % sensitivity equations.
%
%     end

    function do_blur(im, resmap, fixation, source_size)
        % apply resmap to im. Output a picture.
        % Create codecs for grey levels
        c1=sviscodec(im);
        
        % The masks get created when you set the map
        svissetresmap(c1,resmap)
        
        % Encode
        i1=svisencode(c1, fixation(1), fixation(2));
        
        % crop out target area again:
        im_loc = fixation;
        im_loc(2) = im_loc(2) + 430;  % distance of centre from fovea in px.
        
        target = i1(im_loc(1) - source_size(1)/2: im_loc(1) + source_size(1)/2-1,...
            im_loc(2) - source_size(2)/2: im_loc(2) + source_size(2)/2-1);
        
        imwrite(target, sprintf('%s.png', fullfile(out_path, im_id)))
        fprintf('Success!\n\n');
        
    end

end