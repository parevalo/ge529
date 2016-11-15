% This function loads the Amazon LC types for any specified year
% for the analysis window specified in the lab. 

function [LC_Amazon_aggreg] = getLC(year)
    LC_path = '/projectnb/landsat/users/parevalo/ge529/MCD12C1.051/'; % path to your MCD12C1 data

    LC_file_dir = dir(strcat(LC_path,'MCD12C1.A',num2str(year),'*.hdf'));
    LC_file_name = LC_file_dir(1).name;
    land_cover = hdfread(strcat(LC_path,LC_file_name),...
        '/MOD12C1/Data Fields/Majority_Land_Cover_Type_1', 'Index', {[1  1],[1  1],[3600  7200]});
    LC_Amazon = land_cover(1601:2200,2001:2700);
    LC_Amazon_aggreg = uint8(ones(600,700));
    LC_Amazon_aggreg = LC_Amazon_aggreg*4; % non-vegetation
    LC_Amazon_aggreg(LC_Amazon<15 & LC_Amazon~=13)=3; % other vegetation
    LC_Amazon_aggreg(LC_Amazon>7 & LC_Amazon<10)=2; % savannas
    LC_Amazon_aggreg(LC_Amazon>0 & LC_Amazon<6)=1; % forests
    LC_Amazon_aggreg(LC_Amazon==0)=0; % water
end