#RRT* algorithm
import math
import numpy as np
from random import randint
import cv2

# creating usable map
map1=cv2.imread('/content/drive/MyDrive/photos/map_full.png')

mapgrey=cv2.cvtColor(map1,cv2.COLOR_BGR2GRAY)


reqd=mapgrey[:,0]

min=reqd.min()
count=0
for i in reqd:
    if i==min:
         count+=1


m,n=mapgrey.shape
dict1={}

for i in range(m):
    for j in range(n):
        if tuple(map1[i,j]) in dict1:
            dict1[tuple(map1[i,j])]+=1
        else:
            dict1[tuple(map1[i,j])] = 1

print(dict1)
for i in range(2,m-2):
    for j in range(2,n-2):
        lst=[mapgrey[i+1,j] == min and mapgrey[i+2,j] == min,
             mapgrey[i,j+1] == min and mapgrey[i,j+2] == min,
             mapgrey[i-1,j] == min and mapgrey[i-2,j] == min,
             mapgrey[i,j-1] == min and mapgrey[i,j-2] == min]
        count=lst.count(True)
        if count>=2:
            mapgrey[i,j]=min

minima = mapgrey.min()


lst = np.where(mapgrey==minima)
print(lst)
for i,j in zip(lst[0],lst[1]):
    map1[i,j] = np.array([67, 87, 111])
maxima = mapgrey.max()
lstmax = np.where (mapgrey == maxima)
print(lstmax)
for i,j in zip(lstmax[0],lstmax[1]):
    map1[i,j]= np.array([67, 87, 111])
lstmax_x = list(lstmax[0])
lstmax_y = list(lstmax[1])

cv2.imshow(map1)


# unique, counts = np.unique(mapgrey, return_counts=True)

# counts = dict(zip(unique, counts))
# print(counts)




startx=int(lstmax[0].sum()/len(lstmax[0]))
starty=int(lstmax[1].sum()/len(lstmax[1]))
startx,starty
mapgrey[startx,starty] = 255

goalx=int(lst[0].sum()/len(lst[0]))
goaly=int(lst[1].sum()/len(lst[1]))





class line():
    def __init__(self,initial, final):
        self.initial=initial
        self.final=final

        self.length=Node.distance(initial,final)



        self.delx=final.x-initial.x
        self.dely=final.y-initial.y

    def direction(self):
        if self.length == 0:
            return None
        return ((self.xlength)/self.length,(self.ylength)/self.length)


    def pixels(self):
        pixels=[(int(self.initial.x),int(self.initial.y))]
        if math.fabs(self.delx)>math.fabs(self.dely):
            if self.delx<0:
                for i in range(0,int(self.delx),-1):
                    pixels.append((i + self.initial.x,int(i*self.dely/self.delx)+self.initial.y))

            else:
                for i in range(0,int(self.delx)):

                    pixels.append((i + self.initial.x,int(i*self.dely/self.delx)+self.initial.y))
        else:
            if self.dely<0:
                for i in range(0,int(self.dely),-1):
                    pixels.append((int(i*self.delx/self.dely)+self.initial.x,i+self.initial.y))
            else:
                for i in range(0,int(self.dely)):
                    pixels.append((int(i*self.delx/self.dely)+self.initial.x,i+self.initial.y))
        pixels.append((int(self.final.x),int(self.final.y)))
        return pixels



class Node():
    def __init__(self,x,y,parent=None):
        self.x=x
        self.y=y
        self.parent=parent
        if self.parent!= None:
            self.cost=Node.distance(self,parent)
        self.children=[]

    def distance(p1,p2):  #here p1 and p2 are tuples of length 2
        return ((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y))**0.5







class RRTstar:
    def __init__(self,startx,starty,goalx,goaly,map,goalradius,radiusparent,radiuschildren,maxedge,obstraclecolor):

        start=Node(startx,starty,None)
        self.nodes=[start]

        self.start=start
        goal = Node(goalx,goaly,None)
        self.goal=goal
        self.map=map
        self.maxedge=maxedge
        self.goalradius=goalradius
        self.radiusparent=radiusparent
        self.radiuschildren= radiuschildren
        self.obstraclecolor=obstraclecolor

    def addvertex(self,p):
        self.nodes.append(p)

    def nearest(self,p):
        nodes1=self.nodes

        min=Node.distance(p,self.start)
        reqd=self.start
        for node2 in nodes1[1:]:
            dist=Node.distance(node2,p)

            if dist<min:
                reqd=node2
                min=dist


        return reqd


    def reached(self,node1):
        if Node.distance(self.goal,node1)<=self.goalradius:
            return True

    def no_obstacle(self,parent,newnode, DISTANCE = 5):
        connector=line(parent,newnode)
        dist = Node.distance(parent,newnode)
        if dist == 0:
            return False
        pixelslist=connector.pixels()


        for x,y in pixelslist:
            node1=Node(x,y)

            if Node.distance(node1,parent)<=DISTANCE:
                if self.map[x,y][0]!=self.obstraclecolor[0] and self.map[x,y][1]!=self.obstraclecolor[1] and self.map[x,y][2]==self.obstraclecolor[2]:
                    return False
        return True



    def cost(self,node1):
        iter=node1
        cost=0
        while iter.parent!=None:
            cost+=iter.cost
            iter=iter.parent
        return cost

    def createnewnode(self):
        m,n,c=self.map.shape
        i=randint(0,m-1)
        j=randint(0,n-1)
        newnode=Node(i,j,None)



        return newnode

    def findparent(self,p):
        radius = self.radiusparent
        reqd=self.nearest(p)
        possiblex=0
        possibley=0
        min=float('inf')
        if self.no_obstacle(parent = reqd,newnode = p,DISTANCE = self.maxedge):
            dist=Node.distance(reqd,p)
            const=(dist if dist < self.maxedge else self.maxedge)/dist

            possiblex=int((p.x - reqd.x)*const + reqd.x)
            possibley=int((p.y - reqd.y)*const + reqd.y)
            min = self.cost(reqd)
        else:
            reqd=None



        for node1 in self.nodes:

            if Node.distance(node1,p)<=radius:
                if self.no_obstacle(node1,p,DISTANCE = self.radiusparent):


                    cost=self.cost(node1)
                    if cost<min:
                        reqd=node1
                        possiblex=p.x
                        possibley=p.y
        if reqd==None:
            return None
        finalnode=Node(possiblex,possibley,reqd)
        (reqd.children).append(finalnode)


        return finalnode


    def addchildren(self,p):
        costp=self.cost(p)
        for node1 in self.nodes[1:]:
            dist = Node.distance(p,node1)
            if dist<self.radiuschildren and self.no_obstacle(p,node1,DISTANCE = self.radiuschildren):

                costnode = self.cost(node1)
                if costnode > dist + costp:

                    (node1.parent.children).remove(node1)
                    node1.parent = p
                    (p.children).append(node1)



    def showpath(self,path):
        for i in range(len(path)-1):
            line_eqn = line(path[i],path[i+1])
            pixellist = line_eqn.pixels()
            for point in pixellist:
                self.map[point[0],point[1]] = np.array([255,0,0])





    def coloredges(self):
        for node1 in self.nodes:
            for child in node1.children:
                line1 = line(child,node1)
                pixellist = line1.pixels()
                for point in pixellist:
                    self.map[point[0],point[1]] = np.array([0,0,0])

                self.map[pixellist[0][0],pixellist[0][1]] = np.array([255,0,0])
                self.map[pixellist[-1][0],pixellist[-1][1]] = np.array([255,0,0])






    def run(self,interations = 1000):
        result=[]
        for _ in range(interations):
            if _%500 == 0:
                print("iteration -",_)

            node1=self.createnewnode()

            node1 = self.findparent(node1)


            if node1 == None:
                continue


            (self.nodes).append(node1)
            if Node.distance(node1,self.goal) < self.goalradius:
                print('found a new node near the goal')
                result.append(node1)
            self.addchildren(node1)

        self.coloredges()

        cv2.imshow('',self.map)
        
        if len(result) == 0:
            print("NO path found yet")
            return None
        min = self.cost(result[0])
        finalnode = result[0]

        for node1 in result[1:]:
            newcost=self.cost(node1)
            if newcost < min:
                min = newcost
                finalnode = node1

        FINALPATH=[node1]



        while node1.parent != None:
            FINALPATH = [node1.parent] + FINALPATH

            node1 = node1.parent

        self.showpath(FINALPATH)
        cv2.imshow('',self.map)
        return {'path': FINALPATH, 'cost':min}


def showpath(map, path):
    for i in range(len(path)-1):
        line_eqn = line(path[i],path[i+1])
        for point in line_eqn.pixels:
            map[point[0],point[1]] = np.array([0,0,0])

    cv2_imshow(map)

RRTstar1 = RRTstar(startx = 214,starty = 448,goalx = 366,goaly= 492,map = map1,goalradius = 15,radiusparent = 20,radiuschildren = 20,maxedge = 5,obstraclecolor = np.array((11, 27, 47)))

path = RRTstar1.run(1000)