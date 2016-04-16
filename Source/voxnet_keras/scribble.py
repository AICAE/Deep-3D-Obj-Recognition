import os
it = 0
for dirpaths, dirs, fnames  in os.walk('/home/tg/Downloads/volumetric_data'):
    for fname in fnames:
         if fname.endswith('.mat'):
              print(dirpaths)
              print(dirs)
              print(fname)
              it+=1
              if it >= 4:
                   break