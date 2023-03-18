begin
;***********************************************
; get variable names from grib file
;***********************************************
   grib_in  = addfile("./hrrr.t22z.wrfprsf00.grib2","r")   
   names    = getfilevarnames(grib_in); extract all variable names 
   print(names)
end 