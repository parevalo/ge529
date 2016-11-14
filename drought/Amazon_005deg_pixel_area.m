% this is to calculate the area of each 0.05-deg pixel in the Amazon region

resolution = 0.05;
rows = 3600;
cols = 7200;

R = makerefmat(-180+resolution/2, 90-resolution/2, resolution, -resolution);
EP_wt = almanac('earth','wgs84','kilometers');
[~,wt] = areamat(ones(rows,cols,'single'),R,EP_wt);
wt_interest = wt(1601:2200,1);
amazon_cellwt = repmat(wt_interest,[1 700]);

save('amazon_cellwt.mat','amazon_cellwt');

