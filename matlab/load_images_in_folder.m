function ListOfImageNames = load_images_in_folder(folder)
    ListOfImageNames = {};
    if ~isempty(folder)
        if exist(folder,'dir') == false
            msgboxw(['Folder ' folder ' does not exist.']);
            return;
        end
    else
        msgboxw('No folder specified as input for function LoadImageList.');
        return;
    end
    ImageFiles = dir([folder '*.*']);
    
    for Index = 1:length(ImageFiles)
        baseFileName = ImageFiles(Index).name;
        [f, name, extension] = fileparts(baseFileName);
        switch lower(extension)
            case {'.png', '.bmp', '.jpg', '.tif', '.avi'}
            % Allow only PNG, TIF, JPG, or BMP images
            ListOfImageNames = [ListOfImageNames baseFileName];
            otherwise
        end
    end
end