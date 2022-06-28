import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import colorsys
from scipy.spatial import Voronoi
#import wave
from scipy.io.wavfile import read
import utility as ut

class Mod:

    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.width, self.height = self.image.size


    def find_closest(self, X0):
        X1 = []
        for pt in X0:
            r, s = pt
            r0, r1 = int(np.floor(r)), int(np.ceil(r))
            s0, s1 = int(np.floor(s)), int(np.ceil(s))
            all_pts = np.array([[r0, s0], [r0, s1], [r1, s0], [r1, s1]])
            possible_pts = []
            for pt in all_pts:
                if pt[0] < self.width and pt[1] < self.height:
                    possible_pts.append(pt)
            d = []
            for p in possible_pts:
                d.append(np.sqrt((r-p[0])**2 + (s-p[1])**2))
            X1.append(possible_pts[np.argmin(d)])
        X1 = np.array(X1)
        return X1

    def get_colormap(self):
        self.cmap = {}
        for i in range(self.width):
            for j in range(self.height):
                c = np.array(self.image.getpixel((i, j))[:3]) / 255.  
    #            c_ = list(colorsys.rgb_to_hsv(*c))
    #             if c_[0]*360. > 50.:
    #                 c_[1] = 0.
    #             c_[2] = c_[2]**0.4
    #           c__ = np.array(list(colorsys.hsv_to_rgb(*c_)))
                if c.sum() == 0.:
                    c = np.array([1., 1., 1.])
                self.cmap[(i, j)] = c
        return self.cmap


    def find_radius(self, polygon, center):
        num_pts = len(polygon)
        r = []
        for i in range(num_pts):
            p1 = polygon[i]
            if i+1 < num_pts:
                j = i+1
            else:
                j = 0
            p2 = polygon[j]
            r.append(np.linalg.norm(np.cross(p2-p1, p1-center))/np.linalg.norm(p2-p1))
        return min(r)

    def find_circles(self, X0):
        vor = Voronoi(X0)
        X1 = self.find_closest(X0)
        self.x = []
        self.c = []
        self.s = []
    
        for e in range(len(X0)):
            R = vor.regions[vor.point_region[e]]
            if -1 not in R:
                pts = vor.vertices[R]
                x_ = pts[:, 0].sum()/len(R)
                y_ = pts[:, 1].sum()/len(R)
                if x_ > 0. and y_> 0. and x_ < self.width and y_ < self.height:
                    self.x.append([x_ + self.width/2., self.height/2. - y_])
                    self.c.append(self.cmap[(X1[e][0], X1[e][1])])
                    self.s.append(self.find_radius(pts, np.array([x_, y_]))**2)
        
        self.x = np.array(self.x)
        self.c = np.array(self.c)
        self.s = np.array(self.s) 

    def move(self, amplitutde):
        return self.x * (1.0 + amplitutde)

    def plot_steady(self, area_factor=15., marker='$\\bigodot$'):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.scatter(self.x[:, 0], self.x[:, 1], c=self.c, s=area_factor*self.s, marker=marker)
        plt.show() 

    @ut.timer
    def animate(self, wave, animate_as, area_factor=15., marker='$\\bigodot$'):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.axis("off")
        x = self.x
        frame_rate = 24
        move_frames = [i*frame_rate for i in range(int(wave.duration))]
        all_frames = list(range(int(wave.duration) * frame_rate))
        
        def draw_frame(frame_id):
            ax.clear()
            ax.axis("off")
            print('working on frame {}'.format(frame_id))
            a = 0. 
            if frame_id in move_frames:
                i = int(frame_id * wave.samplerate/frame_rate) 
                a = wave.signal[i]
            ax.scatter((1. + a) * self.x[:, 0], (1. + a) * self.x[:, 1], c=self.c, s=area_factor*self.s, marker=marker)
        
        animation = FuncAnimation(fig=fig, func=draw_frame, frames=range(240), interval=100, repeat=False)
        animation.save(animate_as, writer='ffmpeg')


class Wave:
    def __init__(self, sound_path):
        # reading the audio file
        self.samplerate, self.signal = read(sound_path)
        if len(self.signal.shape) > 1:
            self.signal =  self.signal[:, 0] 
        self.signal = np.array(self.signal, dtype=np.float64) / max(self.signal)    
        self.duration = len(self.signal) / self.samplerate
    
        time = np.arange(0, self.duration, 1./self.samplerate)
    
        # using matplotlib to plot
        # creates a new figure
        plt.figure(1)
        
        # title of the plot
        plt.title("Sound Wave")
        
        # label of x-axis
        plt.xlabel("Time")
        
        idx = np.sort(np.random.choice(len(self.signal), replace=False, size=1000))

        # actual plotting
        plt.plot(time[idx], self.signal[idx])
        
        # shows the plot
        # in new window
        plt.show()
    
        # you can also save
        # the plot using
        # plt.savefig('filename')

                  
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    