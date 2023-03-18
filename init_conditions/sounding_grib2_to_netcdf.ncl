begin
;***********************************************
; download grib2 from https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi
;***********************************************
   grib_in  = addfile("./hrrr.t22z.wrfprsf00.grib2","r")   
   names    = (/"PRES_P0_L1_GLC0", "HGT_P0_L1_GLC0", "HGT_P0_L100_GLC0", "POT_P0_L103_GLC0", "TMP_P0_L100_GLC0", "SPFH_P0_L103_GLC0", "SPFH_P0_L100_GLC0", "DPT_P0_L100_GLC0", "UGRD_P0_L100_GLC0", "VGRD_P0_L100_GLC0", "gridlat_0", "gridlon_0"/)
;***********************************************
; create output netcdf file
;*********************************************** 
   system("rm out.nc") ; remove any pre-existing file
   ncdf_out = addfile("out.nc" ,"c")       ; create output netCDF file
;***********************************************
; loop through variables and output each to netcdf
;***********************************************
   do i = 0, dimsizes(names)-1  
   ncdf_out->$names(i)$ = grib_in->$names(i)$
   end do   
end 