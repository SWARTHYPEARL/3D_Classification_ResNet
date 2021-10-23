
import platform

if platform.system() == "Windows":
    from glob import glob
print(glob("./*"))