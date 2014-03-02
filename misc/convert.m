
path = '~/BSR/BSDS500/data/groundTruth/val/'
files = dir(strcat(path , '*.mat'));
%files

for file = files'
    fprintf(' %s \n', file.name)
    load(strcat(path, file.name));
    %groundTruth{1}.Segmentation
    [_, sz] = size(groundTruth)
    for i = 1:sz
        imwrite(
            uint8(getfield(groundTruth{i},'Segmentation')), 
            strcat('~/BSR/BSDS500/data/groundImages/val/', file.name, '_', num2str(i), '.png')
        )
    end
end
