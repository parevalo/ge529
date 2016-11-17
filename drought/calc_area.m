% Function to calculate the area in square kilometers for any given
% pixel size in degrees, for the study area in the Amazon region
% Requires the original raster pixel size resolution and dimensions.
%TODO: Fix function to allow specifying window of interest

function [amazon_cellwt] = calc_area(resolution, rows, cols, outname)
    %resolution = 0.05;
    %rows = 3600;
    %cols = 7200;
    dir = '/projectnb/landsat/users/parevalo/ge529/ge529/drought/data/';
    % lat has to be changed too (90 for modis, 50 for trmm)
    R = makerefmat(-180+resolution/2, 50-resolution/2, resolution, -resolution);
    EP_wt = almanac('earth','wgs84','kilometers');
    [~,wt] = areamat(ones(rows,cols,'single'),R,EP_wt);
    %wt_interest = wt(1601:2200,1);
    wt_interest = wt(161:280,1);
    amazon_cellwt = repmat(wt_interest,[1 140]);
    %amazon_cellwt = repmat(wt_interest,[1 700]);

    save([dir,outname,'.mat'],'amazon_cellwt');
end

%cols_rep: from 400 to 540 (140 cols)*
%rows_rep: from 160 to 280 (120 rows)