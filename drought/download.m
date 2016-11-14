% this is for downloading the MODIS data

clear;
clc;

fid = fopen('data_url_script_2014-11-08_165148.txt');


tline = fgetl(fid);
while ischar(tline)
    disp(tline)
    disp(tline(end-44:end))
    urlwrite(tline,tline(end-44:end));
    tline = fgetl(fid);
end

fclose(fid);