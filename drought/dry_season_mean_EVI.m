% this is to get the dry season EVI at 0.05-degree using MOD13C2

%clear;
%clc;

rows=3600;
cols=7200;

path_MOD13C2 = '/projectnb/landsat/users/parevalo/ge529/MOD13C2/'; % path to your MOD13C2 files

months_non_leap = [182,213,244];
% Create empty array to store annual(quarter) results
annual_vi_cube = single(nan(600,700,10));

for year_i = 3:12
    year_i+1999
    vi_cube = single(nan(rows,cols,3));
    for month_i = 1:3
        file_part = [path_MOD13C2,'MOD13C2.A',num2str(year_i+1999),num2str(months_non_leap(month_i)),'.006.','*.hdf'];
        if (year_i == 5 || year_i == 9)
            file_part = [path_MOD13C2,'MOD13C2.A',num2str(year_i+1999),num2str(months_non_leap(month_i)+1),'.006.','*.hdf'];
        end
        
        MOD13C2_dir = dir(file_part);
        [MOD13C2_file_num,no_use] = size(MOD13C2_dir);
        
        if (MOD13C2_file_num == 0)
            vi=single(nan(rows,cols));
        else
            
        MOD13C2_file = strcat(path_MOD13C2,MOD13C2_dir(1).name);
        qa_vi_mask=nan(rows,cols);
        CMG_0_05_Deg_Monthly_EVI = hdfread(MOD13C2_file,...
            '/MOD_Grid_monthly_CMG_VI/Data Fields/CMG 0.05 Deg Monthly EVI', 'Index', {[1  1],[1  1],[3600  7200]});
        vi = CMG_0_05_Deg_Monthly_EVI;
        vi=single(vi)/10000;
        vi(vi<0 | vi>1)=nan;
        
              
        CMG_0_05_Deg_Monthly_VI_Quality = hdfread(MOD13C2_file,...
            '/MOD_Grid_monthly_CMG_VI/Data Fields/CMG 0.05 Deg Monthly VI Quality', 'Index', {[1  1],[1  1],[3600  7200]});
        qa=CMG_0_05_Deg_Monthly_VI_Quality;
        
        %==============
        qa0_1=2.*bitget(qa,2) + 1.*bitget(qa,1); %bits 0 & 1
        qa2_5=8.*bitget(qa,6) + 4.*bitget(qa,5) + 2.*bitget(qa,4) + 1.*bitget(qa,3); %bits 2-5 VI usefulness
        qa6_7=2.*bitget(qa,8) + 1.*bitget(qa,7); %bits 6-7 aerosol
        qa8=1.*bitget(qa,9); %bit 8 Adjacent cloud detected
        qa10=1.*bitget(qa,11); %bit 10 Mixed clouds
        qa_vi_mask((qa0_1==1 | qa0_1==0) & (qa2_5>=0 & qa2_5<=11) & (qa8==0) & (qa10==0) & (qa6_7>=1 & qa6_7<=2))=1; %MODLAND QA 0 or 1 & VI usefulness [0-11] adj & mixed clouds & shadows & high & clim aerosols off
        clear qa qa0_1 qa2_5 qa6_7 qa8 qa10;
        %==============

        vi(isnan(qa_vi_mask))=NaN;
        
        end
        
        vi_cube(:,:,month_i) = vi;
    end
    
    vi_dry_season = nanmean(vi_cube,3);
    amazon_vi_jas = vi_dry_season(1601:2200,2001:2700);
    annual_vi_cube(:,:,year_i-2) = amazon_vi_jas;
    amazon_vi_jas_file = [path_MOD13C2,'MOD13C2.Amazon.Dry_Season_',num2str(year_i+1999),'.mat'];
    save(amazon_vi_jas_file,'amazon_vi_jas');

end 

% Save cube of annual summer quarter means
save('annual_summer_EVI_means.mat', 'annual_vi_cube')

% Calculate anomalies for 2008 and 2010, removing those years prior to
% calculating the mean and sd on the temporal axis, per pixel
cube_wo2008 = annual_vi_cube;
cube_wo2010 = annual_vi_cube;
cube_wo2008(:,:,7) = NaN; %[];
cube_wo2010(:,:,9) = NaN; %[];
intann_mean_2008 = nanmean(cube_wo2008,3);
intann_mean_2010 = nanmean(cube_wo2010,3);
intann_sd_2008 = nanstd(cube_wo2008, 1, 3);
intann_sd_2010 = nanstd(cube_wo2010, 1, 3);
anomaly_2008 = (annual_vi_cube(:,:,7) - intann_mean_2008)./intann_sd_2008;
anomaly_2010 = (annual_vi_cube(:,:,9) - intann_mean_2010)./intann_sd_2010;

% Plot anomalies (reusing geographic parameters from the other scripts)
anomaly_2008_geo = flipud(anomaly_2008);
anomaly_2010_geo = flipud(anomaly_2010);

Rlatlon = makerefmat('RasterSize', [600 700], ...
        'Latlim', [-20 10], 'Lonlim', [-80 -45]);

geoshow(anomaly_2008_geo, Rlatlon, 'DisplayType', 'surface')
colormap parula
caxis([-2,2])
colorbar

figure
geoshow(anomaly_2010_geo, Rlatlon, 'DisplayType', 'surface')
colormap parula
caxis([-2,2])
colorbar

% Get forest cover per year
LC_08 = getLC(2008);
LC_10 = getLC(2010);

% Histograms of EVI over forest taking areas weighted by pixel area

anomaly_2008(LC_08 ~= 1) = 0;
anomaly_2008(isnan(anomaly_2008)) = 0;
anomaly_2008(anomaly_2008 < -10 | anomaly_2008 > 10) = 0;
a08 = anomaly_2008(anomaly_2008 ~=0);
wt_08 = amazon_cellwt(anomaly_2008 ~=0);
[counts_08, centers_08] = histwc(a08, wt_08 , 500);
bar(centers_08, counts_08, 'EdgeColor', 'blue');

figure
anomaly_2010(LC_10 ~= 1)=0;
anomaly_2010(isnan(anomaly_2010)) = 0;
anomaly_2010(anomaly_2010 < -10 | anomaly_2010 > 10) = 0;
a10 = anomaly_2010(anomaly_2010 ~=0);
wt_10 = amazon_cellwt(anomaly_2010 ~=0);
[counts_10, centers_10] = histwc(a10, wt_10 , 500);
bar(centers_10, counts_10, 'EdgeColor', 'red');


% Save fig
% print(fig,'-dpng','-r600',filename);