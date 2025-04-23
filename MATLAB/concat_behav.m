function [behavior, start_frame, end_frame] = concat_behav(input_frame)
% future improvements: add manual option to input start_frame for beh files

% set(0,'DefaultFigureWindowStyle','docked')
beh_file =uigetfile('*_plane0.mat', 'Chose behavior files to load:','MultiSelect','on');
nFile = size(beh_file,2);
behavior = cell(2,nFile);
frame_count = zeros(nFile+1,1); 
start_frame = zeros(nFile, 1);
end_frame = zeros(nFile+1,1);

if nargin <1   
    for n = 1:nFile
        data = load(beh_file{1,n});
        disp(['loading file: ' beh_file{1,n}])

        % find the incomplete lap before starting to record
        [peaks,loc] = findpeaks(data.behavior.ybinned, 'MinPeakDistance', 100, 'MinPeakProminence',0.1);
        ind = find(loc<2000 & peaks <0.6);
        end_frame(n+1) = end_frame(n)+length(data.behavior.ybinned);
        
        if isempty(ind)
            start_frame(n) = 1+end_frame(n);
            loc0 = 1;
            ind0 = 1;
            disp('warning: behavior file did not start from beginning')
        else    
            % find the start of a new lap
            [~, loc0] = findpeaks(-data.behavior.ybinned, 'MinPeakDistance', 100, 'MinPeakProminence', 0.1);
            ind0 = find(loc0>loc(ind(1)),1);
            start_frame(n) = loc0(ind0)+end_frame(n);
        end

        % check if correct
        figure;plot(data.behavior.ybinned)
        hold on
        scatter(loc0(ind0), data.behavior.ybinned(loc0(ind0)))
        hold off

        behavior{1,n} = data.behavior.ybinned(loc0(ind0):end);
        behavior{2,n} = data.behavior.velocity(loc0(ind0):end);

        frame_count(n+1) = length(data.behavior.ybinned);
    end
    
else    
    % use user input instead
    assert(length(input_frame) == nFile, '# files and start_frame does not matched')
    for n = 1:nFile
        data = load(beh_file{1,n});
        disp(['loading file: ' beh_file{1,n} ' using user input'])
        behavior{1,n} = data.behavior.ybinned(input_frame(n)-sum(frame_count):end);
        behavior{2,n} = data.behavior.velocity(input_frame(n)-sum(frame_count):end);
        frame_count(n+1) = length(data.behavior.ybinned);
        end_frame(n+1) = length(data.behavior.ybinned)+end_frame(n);
    end
    start_frame = input_frame;
end

behavior_mat = cell2mat(behavior);
behavior = struct('ybinned', behavior_mat(1,:),'velocity', behavior_mat(2,:));
% end_frame = cumsum(frame_count);
end_frame = end_frame(2:end);


% name = wildcardPattern(3,4);
% pattern = name + '_plane0.mat';
saveName = extractBefore(beh_file{1,1},'_plane0.mat');
save([saveName '-all-cond.mat'],'behavior','start_frame','end_frame')
