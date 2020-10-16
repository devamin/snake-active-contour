import numpy as np
from numpy import linalg
import math
from scipy import ndimage
import matplotlib.pyplot as plt


class Snake:
    def __init__(self, img, points, alpha=0.015, beta=10, gamma=0.001):
        self.img = img
        self.points = points
        self.contour = np.copy(points)
        self.tempContour = np.copy(self.contour)
        self.alpha = alpha
        self.betas = np.full(points.shape[0], beta)
        self.gamma = gamma
        self.Ecurvs = np.zeros(points.shape[0])
        self.Econts = np.zeros(points.shape[0])
        self.EImgs = np.zeros(points.shape[0])
        self.gradient = self.normalizeGradient(self.getGradient())
        self.distMean = 0
        self.Energy = 0
        self.energyCal()

    def getGradient(self):
        grad_x = ndimage.sobel(self.img, axis=0, mode='constant')
        grad_y = ndimage.sobel(self.img, axis=1, mode='constant')
        grad = np.hypot(grad_x, grad_y).astype(np.uint8)
        return grad

    def normalizeGradient(self, g):
        minimum = g.min()
        maximum = g.max()

        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                g[i][j] = (g[i][j]-minimum)/(minimum-maximum)

        return g

    def getNeighborsMatrix(self, point, nsize=1):
        neighbors = []
        borns = self.img.shape
        for i in range(point[0]-nsize, point[0]+nsize+1):
            for j in range(point[1]-nsize, point[1]+nsize+1):
                if i == point[0] and j == point[1]:
                    continue
                if i >= 0 and i < borns[0]:
                    x = i
                elif i >= borns[0]:
                    continue
                elif i < 0:
                    continue
                if j >= 0 and j < borns[1]:
                    y = j
                elif j >= borns[0]:
                    continue
                elif j < 0:
                    continue
                neighbors.append((x, y))

        return neighbors

    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def dMean(self):
        d = 0
        for i in range(len(self.tempContour)-1):
            d += self.distance(self.tempContour[i], self.tempContour[i+1])
        self.distMean = d/len(self.tempContour)

    def eContPerP(self, i):
        return (self.distMean-linalg.norm(self.tempContour[i]-self.tempContour[i-1]))**2

    def eCurvPerP(self, i):
        i_plus_1 = i+1 if i+1 < self.tempContour.shape[0] else 0
        return linalg.norm(self.tempContour[i-1] - 2*self.tempContour[i] + self.tempContour[i_plus_1])**2

    def energyCurv(self):
        for i in range(0, len(self.tempContour)):
            self.Ecurvs[i] = self.betas[i]*self.eCurvPerP(i)

    def energyImgPerP(self, p):
        return (1/(1+self.gradient[p[0]][p[1]]**2))

    def energyImg(self):
        for i in range(len(self.tempContour)):
            p = self.tempContour[i]
            self.EImgs[i] = self.energyImgPerP(p)

    def energyCont(self, updateFrom=0):
        for i in range(updateFrom, len(self.tempContour)):
            self.Econts[i] = self.eContPerP(i)

    def energyCal(self):
        self.calculEnergies()
        self.Energy = self.energySum()

    def energySum(self):
        E = 0
        E += self.alpha * np.sum(self.Econts)
        E += np.sum(self.Ecurvs)
        E -= self.gamma * np.sum(self.EImgs)
        return E

    def calculEnergies(self):
        self.dMean()
        self.energyCont()
        self.energyCurv()
        self.energyImg()

    def showDif(self):
        fig = plt.figure()
        plt.gray()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cxs, cys = zip(*self.contour.tolist()+[self.contour[0]])
        x, y = zip(*(self.points.tolist()+[self.points[0]]))
        ax1.plot(cxs, cys)
        ax2.plot(x, y)
        ax1.imshow(self.img)
        ax2.imshow(self.img)
        plt.show()

    def run(self, iterations=100, nsize=1):
        it = 0
        while it < iterations:
            it += 1
            print(it)
            self.tempContour = np.copy(self.contour)
            stop = True
            for i in range(0, len(self.tempContour)):
                neighbors = self.getNeighborsMatrix(self.tempContour[i], nsize)
                minimumEnergies = (
                    self.Ecurvs[i], self.EImgs[i], self.Econts[i])

                for n in neighbors:
                    self.tempContour[i] = n
                    self.Ecurvs[i] = self.betas[i] * self.eCurvPerP(i)
                    self.Econts[i] = self.eContPerP(i)
                    self.EImgs[i] = self.energyImgPerP(n)
                    tempEnergy = self.energySum()
                    if(self.Energy > tempEnergy):
                        print("\t", i, "from",
                              self.contour[i], "to", n, "energy", tempEnergy)

                        # move to the new point with the lowest energy
                        self.contour[i] = n
                        self.Energy = tempEnergy

                        minimumEnergies = (
                            self.Ecurvs[i], self.EImgs[i], self.Econts[i])
                        stop = False
                    else:
                        self.Ecurvs[i], self.EImgs[i], self.Econts[i] = minimumEnergies

                # update Energies and mean distance
                self.calculEnergies()

            # set betas to 0 for the maximum Ecurvs point
            self.betas[np.argmax(self.Ecurvs)] = 0

            # no new change to the snake minimum local
            if stop:
                break
