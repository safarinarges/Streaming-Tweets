import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use("ggplot")

figures = plt.figure()
axis = figures.add_subplot (1,1,1)

def anim(i):
    pulldata = open ("twitter-out.txt", "r").read()
    lines = pulldata.split('\n')
    
    xarray = []
    yarray = []
    
    x = 0
    y = 0
    
    for l in lines:
        x+=1
        if "pos" in l:
            y+=1
        elif "neg" in l:
            y-=1
            
        xarray.append(x)
        yarray.append(y)
        
        axis.clear()
        axis.plot(xarray,yarray)
animate = animation.FuncAnimation(figures, anim, intreval=1000)
plt.show()    

        