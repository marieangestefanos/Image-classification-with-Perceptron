function printimg(img)


close all

image(img)
colormap(flipud(gray(255)));
caxis([0 1]);
colorbar;
