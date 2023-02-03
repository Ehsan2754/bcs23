import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pltPlanesIn3D(norms=[], offsets=[], points=[], xyranges=[10, 10], alpha=0.6, title='', pltRow=False,figsize=(16,10),dpi=160):

    if not len(norms) == len(offsets):
        raise ValueError(
            'offset vector length should be equal to the norms length.')

    # Create the figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Add an axes
    if not pltRow:
        ax = fig.add_subplot(111, projection='3d')
        for point_tensor in points:
            for point in point_tensor:
                # and plot the points
                ax.scatter(point[0], point[1], point[2])


    # Sample
    # print (max(norms),min(norms))
    xx, yy = np.meshgrid(range(-xyranges[0],xyranges[0]), range(-xyranges[1],xyranges[1]))

    for idx, normal in enumerate(norms):

        if pltRow:
            ax = plt.subplot(1, len(norms), idx+1, projection='3d')
            for point in points[idx]:
                ax.scatter(point[0], point[1], point[2])

        # a plane is a*x+b*y+c*z+d=0
        # [a,b,c] is the normal. Thus, we have to calculate
        a, b, c = normal

        d = offsets[idx]

        zz = (a * xx + b * yy + d) * -1. / c

        # plot the surfaces

        ax.plot_surface(xx, yy, zz, alpha=alpha)
    plt.title(title)
    plt.show()



# produces random plane norm vectors in  a normal distribution 
def getRandom3DPlaneNorms(n, mean = 0, std = 1,offsets=False):
    return (torch.normal(mean,std,size=(n, 3),dtype=torch.double), torch.rand(size=(n, 1),dtype=torch.double)) if offsets \
        else (torch.normal(mean,std,size=(n, 3),dtype=torch.double), torch.zeros(n,1,dtype=torch.double))

# gives samples of  a plane specifying the range, mean and standard deviation of the samples.
def sample3DPlane(n,norms,offsets,xyrange=[4,4],mean = 0, std = 1):
    pts = []
    for idx,norm in enumerate(norms):
        a,b,c = norm
        
        xx = (torch.normal(mean=mean,std=std,size=(n,),dtype=torch.double).T)*xyrange[0]
        yy = (torch.normal(mean=mean,std=std,size=(n,),dtype=torch.double).T)*xyrange[1]
        zz = (a * xx + b * yy + offsets[idx]) * -1. / c
        pts.append(torch.column_stack((xx,yy,zz)))
    return torch.stack(pts)