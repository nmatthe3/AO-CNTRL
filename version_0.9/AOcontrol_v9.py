# -*- coding: utf-8 -*-
"""
Version created 2021-06-10

@author: Nolan Matthews
"""

import sys,serial,time,os
from PyQt5.QtWidgets import QApplication,QWidget,QSlider
from PyQt5.QtCore import QObject,QThread,pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5 import QtCore,uic
import datetime
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import center_of_mass
import zwoasi as asi
from lmfit import Parameters,minimize#,report_fit

class EmittingStream(QObject):
    
    textWritten = pyqtSignal(str)
    
    def write(self,text):
        self.textWritten.emit(str(text))
        return 
    
def twoD_GAUSS(x,y,A,x0,y0,sig_x,sig_y,theta,offset):
    '''Returns 2D Gauss Function used to evaluate centroid/PSF'''
    xo = float(x0)
    yo = float(y0)
    a = np.cos(theta)**2 / (2*sig_x**2) + np.sin(theta)**2 / (2*sig_y**2)
    b = -np.sin(2*theta) / (4*sig_x**2) + np.sin(2*theta) / (4*sig_y**2)
    c = np.sin(theta)**2 / (2*sig_x**2) + np.cos(theta)**2 / (2*sig_y**2)
    g_2d = offset + A * np.exp(-(a*(x-xo)**2 + 2*b*(x-xo)*(y-yo) + c*(y-yo)**2))
    return g_2d

class AOunit(QObject):
    """
    Class to communicate with the adaptive optics unit / ZWO ASI camera
    """
    finished = pyqtSignal()

    def __init__(self):
        super(AOunit,self).__init__()

        self.port_address = 'COM5'
        
        try:
            self.ser = serial.Serial(self.port_address)
        except:
            print("Connection to AO device failed... check COM number")
            
        #List of serial commands for adapative optics unit 
        self.cmd_center = b"K      "
        self.cmd_north = b"GN00001"
        self.cmd_south = b"GS00001"
        self.cmd_east  = b"GT00001"
        self.cmd_west  = b"GW00001"
        self.cmd_center_slow = b"R      "
        self.cmd_mount_north = b"MN00001"
        self.cmd_mount_south = b"MS00001"
        self.cmd_mount_east = b"MT00001"
        self.cmd_mount_west = b"MW00001"
        
        #
        self.mount_time_limit = 5  

        #Initialization of movement parameters/variables 
        self.raster_steps = 50 #range of raster scan steps
        self.step_limit = 40 #total step limit from center for the AO unit
        self.shouldWait = False #choose whether to implement wait after AO cmd. 
        self.wait_time = 0.250
        
        #seconds to wait after each serial write
        self.n_clicks = 0 #number of times the position has been adjusted 
        self.north_south_clicks = 0 #tracker for # of steps in N-S direction
        self.east_west_clicks = 0 #tracker for # of steps in E-W direction
        #self.move_center() #initialize center position. 
        
        #Initialization of control algorithm parameters
        self.circle_of_alignment_radii = 1 #radial number of pixels in which corrections are not needed.
        self._x_CA = 0 #optimal x-position
        self._y_CA = 0 #optimal y-position
        self.controlAlgorithmIsRunning = False
        self.signalThreshold = 5 
        
        self.Kp = 1
        self.Ki = 0
        self.Ti = 10
        
        #Define field rotation params
        self.field_rotation_ang = 0.0 #degrees
        self.use_field_rotation = True
        self.quad1 = [315,45]
        self.quad2 = [45,135]
        self.quad3 = [135,225]
        self.quad4 = [225,315]
        
        #======================ZWO ASI CAMERA CONTROL=========================
        #
        #
        #Since the control algorithm directly calls the CCD for acquiring 
        #images it is included within this class 'AOunit such that the 
        #parameters can be directly modified by the GUI. Perhaps there is a 
        #better way to do this, ideally the AO / CCD are in different classes. 
        #
        #
        
        #Path to .dll file for control of ZWO ASI CCD
        self.asi_lib_path = "C:/Users/Nolan/Downloads/ASI_Windows_SDK_V1.18/ASI_SDK/lib/x64/ASICamera2.dll"
        
        #Initialize the ASI binding                        
        try:
            asi.init(library_file = self.asi_lib_path)
        except:
            print("ASI Initializiation Skipped or Failed")
            pass
        
        #Create instance of camera (assumes only one camera is connected)
        self.camera = asi.Camera(0) #Create instance of camera
        
        #Set image type
        self.camera.set_image_type(asi.ASI_IMG_RAW8)
        
        #Set default camera settings
        self.camera.max_height = self.camera.get_camera_property()['MaxHeight']
        self.camera.max_width = self.camera.get_camera_property()['MaxWidth']
        self.camera.set_control_value(asi.ASI_GAIN,1)
        self.camera.set_control_value(asi.ASI_EXPOSURE,1000)
        self.camera.set_roi(start_x=0,start_y=0,
                            height=self.camera.max_height,
                            width=self.camera.max_width
                            )
        
        self.noise_level = 1
        
        #Turn on 'high speed mode' (not sure if this has an effect in video mode)
        self.camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,1)
        
        #Set Image Type ==> bit resolution can be left at 8-bit
        self.camera.set_image_type(asi.ASI_IMG_RAW8)
        
        #Start 'video' mode for fast captures
        self.camera.start_video_capture()
        
        #Capture initial frame
        self.flip_image_x = False
        self.flip_image_y = False
        self.image_data = self.camera.capture_video_frame()
        
        #Controls for analyzing images
        self.analyzeEachFrame = False        
        self.fit_report = ''
        
        #Variables for streaming images
        self.recordImages = False 
        self.recordCounter = 0
        self.recordFrames = 10
        return

    def grab_image(self):
        '''Helper function that grabs a CCD frame and applies mirroring'''
        raw_image = self.camera.capture_video_frame()
        
        #Set threshold for noise values as to not affect centroid measurements
        indxs_noise = raw_image < self.noise_level
        raw_image[indxs_noise] = 0.0
        
        if self.flip_image_x == True:
            raw_image = np.flip(raw_image,axis=1)
        if self.flip_image_y == True:
            raw_image = np.flip(raw_image,axis=0)
            
        #----------Analyze each frame in detail----------
        if self.analyzeEachFrame == True:
            try:
                self.perform2DgaussFit()
            except:
                pass
            #self.analyzeFrame(self.image_data)
            
        
        #----------IF RECORD IS ON, STORE ARRAY----------
        if self.recordImages == True:
            if self.recordCounter == 0:
                #Initialize 3d array to be saved on first iteration
                ny,nx = np.shape(raw_image)
                self.frame_array = np.zeros(shape=(self.recordFrames,ny,nx),dtype='int8')
                self.t_start = time.time()
                self.t_elapsed = np.zeros(self.recordFrames)
            if self.recordCounter >= 0:
                #Record image
                self.frame_array[self.recordCounter] = raw_image
                self.t_elapsed[self.recordCounter] = time.time() - self.t_start
                self.recordCounter +=1
            if self.recordCounter == self.recordFrames:
                #Perform reset of counter and turn off streaming 
                self.recordImages = False #turn off streaming
                self.recordCounter = 0 #reset counter
                self.saveImageArray()
                #TODO: Perform save of array
        else:
            pass
                
        return raw_image
    
    def initSaveDir(self):
        
        imagepath = './'
        
        today = datetime.date.today()
        today_fmtd = today.strftime("%Y_%m_%d")

        date_path = os.path.join(imagepath,today_fmtd)
        if not os.path.exists(date_path):
            os.mkdir(date_path)
        
        dir_doesnt_exist,ii=True,0
        while dir_doesnt_exist:
            potential_savedir = os.path.join(date_path,str(ii))
            if not os.path.exists(potential_savedir):
                os.mkdir(potential_savedir)
                self.imageSavePath = potential_savedir
                dir_doesnt_exist = False
            else:
                ii+=1
        
        return        
    
    def saveImageArray(self):
        
        self.initSaveDir()
        
        print("Saving Frames in Image Array")
        nIm,nx,ny = np.shape(self.frame_array)
        for ii in range(nIm):
            fileout = self.imageSavePath+'/CCDimage_%i' % ii
            t_elapsed_str = "T_elapsed (s): %s\nIN CONTROL LOOP: %s"%(str(self.t_elapsed[ii]),
                                                                            str(self.controlAlgorithmIsRunning))
            np.savetxt(fileout,
                       self.frame_array[ii],
                       delimiter='\t',
                       header = t_elapsed_str,
                       comments = '',
                       fmt='%i')

    #-----------CONTROL ALGORITHM------------
    def start_control_algorithm(self):
        self.controlAlgorithmIsRunning = True
        self.run_control_algorithm
        return
    
    def stop_control_algorithm(self):
        print("Stopping Control Algorithm...")
        self.controlAlgorithmIsRunning = False
        return
    
    def run_control_algorithm(self):
        """
        The main function for the control algorithm of the adaptive optics unit
        """
        print("Starting Control Algorithm...")
        self.controlAlgorithmIsRunning = True
        
        e_x_hist = np.array([])
        e_y_hist = np.array([])
        
        while self.controlAlgorithmIsRunning:
            self.image_data = self.grab_image() #get processed image
            [yc,xc] = center_of_mass(self.image_data) #get centroid 
            
            #Calculate difference between optimal and centroid positions
            del_x,del_y = xc - self._x_CA, yc - self._y_CA
            
            #-----------PI CONTROL-------------------
            if len(e_x_hist) == self.Ti:
                e_x_hist = np.delete(e_x_hist,0)
                e_y_hist = np.delete(e_y_hist,0)

            e_x_hist = np.append(e_x_hist,del_x)
            e_y_hist = np.append(e_y_hist,del_y)
            u_x = self.Kp * del_x + self.Ki / self.Ti * e_x_hist.sum()
            u_y = self.Kp * del_y + self.Ki / self.Ti * e_y_hist.sum()

            N_x = round(u_x) #number of steps to move E/W
            N_y = round(u_y) #number of steps to move N/S
            
            in_limit_x = np.abs(self.east_west_clicks) < self.step_limit
            in_limit_y = np.abs(self.north_south_clicks) < self.step_limit
            
            limit_x_E = self.east_west_clicks == self.step_limit
            limit_x_W = self.east_west_clicks == -self.step_limit
            limit_y_N = self.north_south_clicks == self.step_limit
            limit_y_S = self.north_south_clicks == -self.step_limit
            
            #Ensure that there is a signal before making corrections
            signal_found = self.image_data.max() > self.signalThreshold
            
            #THIS OVERRIDES THE PI CONTROL AND DOES 1:1 MOVEMENT
            #N_x = del_x
            #N_y = del_y
            
            #if signal_found:
            #print("SIGNAL NOT FOUND---> NO CORRECTIONS SENT!!!")
            
            if signal_found:
    
                
                #Check if tip-tilt is at stroke limit, if so,
                #push it back into it's working range. 
                if limit_x_E:
                    self.move_west()
                if limit_x_W:
                    self.move_east()
                if limit_y_N:
                    self.move_south()
                if limit_y_S:
                    self.move_north()
                
                if N_x > 0:
                    if in_limit_x:
                        self.move_west()
                    if limit_x_W:
                        print("Beyond -X limit..")
                        self.move_mount_west_mod()
                        self.wait()
                        self.move_east()
                if N_x < 0:
                    if in_limit_x:
                        self.move_east()
                    if limit_x_E:
                        print("Beyond +X limit..")
                        self.move_mount_east_mod()
                        self.wait()
                        self.move_west()
                if N_y > 0:
                    if in_limit_y:
                        self.move_south()
                    if limit_y_S:
                        print("Beyond +Y Limit")
                        self.move_mount_north_mod()
                        self.wait()
                        self.move_north()
                if N_y < 0: 
                    if in_limit_y:
                        self.move_north()
                    if limit_y_N:
                        print("Beyond -Y Limit")
                        self.move_mount_south_mod()
                        #self.move_mount_north()
                        self.wait()
                        self.move_south()
                
        return

    #------------IMAGE ANALYSIS---------------
    
    def perform2DgaussFit(self):
        
            #Extract (x,y) meshgrid based on size of image
            n_y_pixels,n_x_pixels = np.shape(self.image_data)
            xdat = np.linspace(0,n_x_pixels,n_x_pixels)
            ydat = np.linspace(0,n_y_pixels,n_y_pixels)
            x,y = np.meshgrid(xdat,ydat)
    
            #Setup initial paramters for the fit
            initial = Parameters()
            initial.add("amp",value=self.image_data.max(),min=0)
            initial.add("centroid_x",value = n_x_pixels/2)
            initial.add("centroid_y",value = n_y_pixels/2)
            initial.add("sigma_x",value = 3,min=0.1)
            initial.add("sigma_y",value = 3,min=0.1)
            initial.add("theta",value = 0.01,min=0.0,max=np.pi/4)
            initial.add("offset",value = 2)
    
            #Run the fit
            self.fit = minimize(self.residuals,initial,args=(x,y,self.image_data))
            
            #Store fit results
            self.fit_report = self.make_fit_report()
        
            return
        
    def make_fit_report(self):
        
        #Create string buffer for output
        buff = []
        add = buff.append
        
        #Iterate through parameters, get values/uncertainties
        params = self.fit.params
        parnames = list(params.keys())
        for name in parnames:
            par = params[name]
            try:
                line = "%s: %0.3f +/- %0.3f" % (name,par.value,par.stderr)
            except:
                line = "fit failed..."
            add(line)
        
        return '\n'.join(buff)
    
    def residuals(self,p,x,y,z):
        A = p["amp"].value
        cen_x = p["centroid_x"].value
        cen_y = p["centroid_y"].value
        sig_x,sig_y = p["sigma_x"].value,p["sigma_y"].value
        offset = p["offset"].value
        theta = p["theta"].value

        return (z - twoD_GAUSS(x,y,A,cen_x,cen_y,sig_x,sig_y,theta,offset))
    

    #-----------AUXILLARY FUNCTIONS------------------
    
    
    def check_limit(self):
        """
        Checks if the AO limit is beyond limit switch
        """
        if self.ser.read(1) == b'L':
            print("Limit Switch Reached!!! Consider realignment with the mount")
        return

    def print_position(self):
        print('N-S: %i, E-W: %i' % (self.north_south_clicks,self.east_west_clicks))
        return

    def wait(self):
        if self.shouldWait:
            time.sleep(self.wait_time)
        return

    #--------------MOVEMENT FUNCTIONS-----------------
    def move_north(self):
        self.ser.write(self.cmd_north)
        self.check_limit()
        self.north_south_clicks += 1
        self.n_clicks += 1
        self.print_position()
        self.wait()
        return
    
    def move_east(self):
        self.ser.write(self.cmd_east)
        self.check_limit()
        self.n_clicks += 1
        self.east_west_clicks += 1
        self.print_position()
        self.wait()
        return
    
    def move_south(self):
        self.ser.write(self.cmd_south)
        self.check_limit()
        self.n_clicks += 1
        self.north_south_clicks -= 1
        self.print_position()
        self.wait()
        return
    
    def move_west(self):
        self.ser.write(self.cmd_west)
        self.check_limit()
        self.n_clicks += 1
        self.east_west_clicks -= 1
        self.print_position()
        self.wait()
        return
    
    def move_center(self):
        self.ser.write(self.cmd_center)
        self.n_clicks += 1
        self.north_south_clicks = 0
        self.east_west_clicks = 0
        self.print_position()
        return
    
    def move_center_slow(self):
        self.ser.write(self.cmd_center_slow)
        self.n_clicks += 1
        self.north_south_clicks = 0
        self.east_west_clicks = 0
        self.print_position()
        return
    
    def get_field_angle_quadrant(self):
        if self.use_field_rotation == False:
            return 1
        if (self.field_rotation_ang > self.quad1[0]) or (self.field_rotation_ang <= self.quad1[1]):
            return 1
        if (self.field_rotation_ang > self.quad2[0]) and (self.field_rotation_ang <=self.quad2[1]):
            return 2
        if (self.field_rotation_ang > self.quad3[0]) and (self.field_rotation_ang <=self.quad3[1]):
            return 3
        if (self.field_rotation_ang > self.quad4[0]) and (self.field_rotation_ang <=self.quad4[1]):
            return 4
    
    def move_mount_north_mod(self):
        quad = self.get_field_angle_quadrant()
        if quad == 1:
            self.move_mount_north()
        if quad == 2:
            self.move_mount_east()
        if quad == 3:
            self.move_mount_south()
        if quad == 4:
            self.move_mount_west()

    def move_mount_south_mod(self):
        quad = self.get_field_angle_quadrant()
        if quad == 1:
            self.move_mount_south()
        if quad == 2:
            self.move_mount_west()
        if quad == 3:
            self.move_mount_north()
        if quad == 4:
            self.move_mount_east()

    def move_mount_east_mod(self):
        quad = self.get_field_angle_quadrant()
        if quad == 1:
            self.move_mount_east()
        if quad == 2:
            self.move_mount_south()
        if quad == 3:
            self.move_mount_west()
        if quad == 4:
            self.move_mount_north()

    def move_mount_west_mod(self):
        quad = self.get_field_angle_quadrant()
        if quad == 1:
            self.move_mount_west()
        if quad == 2:
            self.move_mount_north()
        if quad == 3:
            self.move_mount_east()
        if quad == 4:
            self.move_mount_south()

    def move_mount_north(self):
        self.ser.write(self.cmd_mount_north)
        print("Sending Mount Command - North")
        #self.wait()
        
    def move_mount_south(self):
        self.ser.write(self.cmd_mount_south)
        print("Sending Mount Command - South")
        #self.wait()

    def move_mount_east(self):
        self.ser.write(self.cmd_mount_east)
        print("Sending Mount Command - East")
        #self.wait()
        
    def move_mount_west(self):
        self.ser.write(self.cmd_mount_west)
        print("Sending Mount Command - West")
        #self.wait()
    
    def scan_east(self):
        for ii in range(self.raster_steps):
            self.move_east()
        return
    
    def scan_west(self):
        for ii in range(self.raster_steps):
            self.move_west()
        return

    def perform_raster_scan(self):
        print("Performing Raster Scan...")

        #Initialize position before scan at N-W corner
        for ii in range(int(self.raster_steps/2)):
            self.move_north()
            self.move_west()

        #Run raster scan...
        for ii in range(int(self.raster_steps/2)):
            self.scan_east()
            self.move_south()
            self.scan_west()
            self.move_south()
        return


class AO_interface(QWidget):
    """
    Class that constructs the adaptive optics control GUI
    """

    def __init__(self):
        super(AO_interface,self).__init__()

        #Load User Interface
        uic.loadUi('aoUI_v0.9.ui',self)
        
        self.setWindowTitle("Adaptive Optics Control v0.8")
        self.AO = AOunit() #reference the AO unit class as AO 
        self.init_UI() #build user interface

    def update_pos_plot_data(self):
        self.pos_line.setData([self.AO.east_west_clicks],[self.AO.north_south_clicks])

    def update_histogram_data(self):
        hist_image_raveled = np.array(self.AO.image_data).ravel()
        image_raveled = hist_image_raveled[hist_image_raveled > self.hist_x_min]
        hist,bin_edge = np.histogram(image_raveled,bins=self.hist_bins)
        hist_norm = hist / np.sum(hist)
        
        bin_centers = (bin_edge[:-1] + bin_edge[1:])/2
        self.histogram_line.setData(bin_centers,hist_norm)

    def update_ccd_image(self):
        if not self.AO.controlAlgorithmIsRunning:
            #If CA is running, then self.image_data is alrady updating
            self.AO.image_data = self.AO.grab_image()
        self.im_item.setImage(self.AO.image_data.T)
        
    def make_box(self):
        x,y = np.array([]),np.array([])
        l = self.AO.step_limit
        arr = np.arange(-self.AO.step_limit,self.AO.step_limit)
        #Verticle line on the left
        x = np.append(x,np.ones(2*l) * -self.AO.step_limit)
        y = np.append(y,arr)
        #Horizontal line on the top
        x = np.append(x,arr)
        y = np.append(y,np.ones(2*l) * self.AO.step_limit)
        #Verticle line on the right
        x = np.append(x,np.ones(2*l) * self.AO.step_limit)
        y = np.append(y,arr[::-1])
        #Horizontal line on the bottom
        x = np.append(x,arr[::-1])
        y = np.append(y,np.ones(2*l) * -self.AO.step_limit)
        return [x,y]

    def draw_circle(self,x0,y0):
        thet_vals = np.arange(0,2*np.pi,0.001)
        r_val = 13.0
        return [x0+r_val * np.sin(thet_vals),y0+r_val * np.cos(thet_vals)]
        
    def output_terminal_written(self,text):
        #cursor = QTextCursor(self.output_terminal_textEdit.document())
        #cursor.setPosition(0)
        #self.output_terminal_textEdit.setTextCursor(0)
        self.output_terminal_textEdit.insertPlainText(text)
        self.output_terminal_textEdit.verticalScrollBar().setValue(self.output_terminal_textEdit.verticalScrollBar().maximum())
     
    def init_UI(self):
        '''
        Creates the GUI and connection between PyQT5 widgets and AO actions
        '''
        #----Set up text log-----------
        #Connect sys.stdout to instance of EmittingStream
        sys.stdout = EmittingStream(textWritten=self.output_terminal_written)
        #sys.stdout = EmittingStream(textWritten=self.output_terminal_written)
        
        #----Initialize location of field-rotation angle file:
        self.field_rotation_filepath = "C:/Users/Nolan/Documents/Research/I2C/AO/FieldRotationAngleFile/Tool_Derotator_T1M_FieldRotationAngle_Current.txt"
        self.invert_field_rot_angle = False
        
        #----Initialize tip-tilt position plot----
        self._x_pos = [0] #init x-position
        self._y_pos = [0] #init y-position
        self.position_plot.setBackground('w') #set background color to white
        self.pos_line = self.position_plot.plot([self.AO.east_west_clicks],[self.AO.north_south_clicks],symbol='+')
        box_pen = pg.mkPen(color=(255,0,0),width=5) #set border linestyle
        self.position_plot.plot(self.make_box()[0],self.make_box()[1],pen=box_pen) #make border
        self.position_plot.setTitle("Tip-Tilt Position")
        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update_pos_plot_data)
        self.timer.start()

        self.position_plot.setXRange(-self.AO.step_limit,self.AO.step_limit)
        self.position_plot.setYRange(-self.AO.step_limit,self.AO.step_limit)        

        #Create timer for loading field-rotation angle: 
        self.timer_FRA = QtCore.QTimer()
        self.timer_FRA.setInterval(2000) #in ms
        self.timer_FRA.timeout.connect(self.update_field_rotation_angle)
        self.timer_FRA.start()

        #----Initialize window for viewing the CCD image-----
        self.image_data = np.array([])
        self.im_item = pg.ImageItem()
        self.im_item.scale(1,1)
        self.im = self.glw.addPlot()
        self.im.addItem(self.im_item)

        init_im = np.random.rand(1280,960)
        self.im_item.setImage(init_im)

        x_circ,y_circ = self.draw_circle(x0=1280/2,y0=960/2)
        self.circ_psf = self.im.plot(x_circ,y_circ)
        self.circ_setpoint_h = self.im.plot(x_circ,y_circ,pen='r')
        self.circ_setpoint_v = self.im.plot(x_circ,y_circ,pen='r')

        self.ROI_size = 96 #size of extracted ROI (width/height)
        
        #-----Initialize cross-sections plot-----
        self.cross_sections.setBackground('w')
        self.cross_sections.setXRange(0,self.ROI_size)
        
        self.x_cross_section = self.cross_sections.plot([],[])
        self.y_cross_section = self.cross_sections.plot([],[])
   
    
        #----Initialize histogram plot-------------
        self.histogram_plot.setBackground('w')
        self.histogram_line = self.histogram_plot.plot([],[])
        self.timer.timeout.connect(self.update_histogram_data)
        self.hist_x_min,self.hist_x_max = 10,256
        self.histogram_plot.setXRange(self.hist_x_min,self.hist_x_max)
        #self.histogram_plot.setYRange(0,1)
        self.hist_bins = np.arange(self.hist_x_min,self.hist_x_max)
        
        self.timer.timeout.connect(self.update_ccd_image)
        self.timer.timeout.connect(self.get_centroid)
        
        #Create labels for control algorithm optimization
        self._update_opt_B.clicked.connect(self._update_opt_)
        
        #Create connection between button/action
        #Connect manual control buttons
        self.north.clicked.connect(self.AO.move_north)
        self.east.clicked.connect(self.AO.move_east)
        self.south.clicked.connect(self.AO.move_south)
        self.west.clicked.connect(self.AO.move_west)
        self.center.clicked.connect(self.AO.move_center)
        #Connect raster scan buttons (deprecated..)
        self.raster_scan.clicked.connect(self.threaded_raster_scan)
        self.center_slow.clicked.connect(self.AO.move_center_slow)
        self.close_button.clicked.connect(self.close_application)
        self.updateCameraSettingsB.clicked.connect(self._update_camera_settings)
        self.extractROIB.clicked.connect(self.extractROI)
        self.extAboutCent_B.clicked.connect(self.extractAboutCent_action)
        #Connect optimal position correction buttons
        self._opt_pos_upB.clicked.connect(self._opt_pos_up)
        self._opt_pos_downB.clicked.connect(self._opt_pos_down)
        self._opt_pos_rightB.clicked.connect(self._opt_pos_right)
        self._opt_pos_leftB.clicked.connect(self._opt_pos_left)
        self.update_about_centroid_B.clicked.connect(self.update_about_centroid_action)
        #Connect mount control buttons
        self.mount_N_B.clicked.connect(self.AO.move_mount_north_mod)
        self.mount_S_B.clicked.connect(self.AO.move_mount_south_mod)
        self.mount_E_B.clicked.connect(self.AO.move_mount_east_mod)
        self.mount_W_B.clicked.connect(self.AO.move_mount_west_mod)
        self.updateCMDduration_B.clicked.connect(self.updateMountCMD)
        #Connect image processing buttons
        self.flip_x_btn.setChecked(self.AO.flip_image_x)
        self.flip_y_btn.setChecked(self.AO.flip_image_y)
        self.flip_x_btn.stateChanged.connect(self.flip_x_action)
        self.flip_y_btn.stateChanged.connect(self.flip_y_action)
        self.get_full_resolution_B.clicked.connect(self.get_full_resolution)
        self.angular_units_B.clicked.connect(self.change_image_units)
        #Connect field rotation buttons
        self.use_field_rot_B.setChecked(self.AO.use_field_rotation)
        self.use_field_rot_B.clicked.connect(self.set_use_of_field_rotation)
        self.invert_field_rot_B.clicked.connect(self.set_field_rotation_direction)

        self.analyzeFrame_B.clicked.connect(self.analyzeFrame_action)
        self.saveFrames_B.clicked.connect(self.saveFrames_action)

        #----------
        self.signalThreshold_L.setText(str(self.AO.signalThreshold))

        #Connect PI update button
        self.updatePIvalues_B.clicked.connect(self.updatePIvalues_action)
        self.propValue_slider.setMinimum(0)
        self.propValue_slider.setMaximum(1000)
        self.propValue_slider.setValue(100)
        self.propValue_slider.setTickPosition(QSlider.TicksAbove)
        self.propValue_slider.setSingleStep(10)
        self.propValue_slider.setTickInterval(100)
        self.intValue_slider.setMinimum(0)
        self.intValue_slider.setMaximum(200)
        self.intValue_slider.setValue(50)
        self.intValue_slider.setTickPosition(QSlider.TicksAbove)
        self.intValue_slider.setSingleStep(10)
        self.intValue_slider.setTickInterval(10)
        
        self.propValue_slider.valueChanged.connect(self.updatePIvalues_action)
        self.intValue_slider.valueChanged.connect(self.updatePIvalues_action)
        self.updatePIvalues_action()
        
        #Place AO object in own thread / make connections to buttons:
        #Create new thread so GUI does not freeze
        self.thread = QThread()
        self.thread.start()

        #Move AO object to thread
        self.AO.moveToThread(self.thread)

        #Connect signals and slots???
        #self.thread.started.connect(self.AO.run_control_algorithm)
        
        #self.run_CA_B.clicked.connect(self.threaded_control_algorithm)
        self.run_CA_B.clicked.connect(self.AO.run_control_algorithm)
        self.run_CA_B.clicked.connect(self.set_stop_CA_button)
        
        self.stop_CA_B.clicked.connect(lambda: self.AO.stop_control_algorithm())
        self.stop_CA_B.clicked.connect(self.set_start_CA_button)
        self.stop_CA_B.setEnabled(False)

        self.AO.finished.connect(self.thread.quit)        
        return        
    
    def updateMountCMD(self):
        CMD_value = str(self.CMD_duration_L.text())
        print("Updating mount duration to %s ms" % CMD_value)
        num = CMD_value.zfill(5)
        self.AO.cmd_mount_north = bytes("MN"+num,"utf-8")
        self.AO.cmd_mount_south = bytes("MS"+num,"utf-8")
        self.AO.cmd_mount_east = bytes("MT"+num,"utf-8")
        self.AO.cmd_mount_west = bytes("MW"+num,"utf-8")
        return

    def change_image_units():
        return
    
    def updatePIvalues_action(self):
        Kp = self.propValue_slider.value() / 1000
        Ki = self.intValue_slider.value() / 100
        self.propTerm_L.setText(str(Kp))
        self.intTerm_L.setText(str(Ki))
        self.AO.Ki = float(Kp)
        self.AO.Kp = float(Ki)
    
    def saveFrames_action(self):
        self.AO.recordFrames = int(self.nFrames_L.text())
        self.AO.recordImages = True
        return
    
    def analyzeFrame_action(self):
        self.AO.perform2DgaussFit()
        self.updateFitResults()
        self.updateCrossSectionPlot()
        
    def updateCrossSectionPlot(self):
        hor_vals = np.arange(self.ROI_size)
        x_cent = self.AO.fit.params["centroid_x"].value
        y_cent = self.AO.fit.params["centroid_y"].value
        x_cross_vals = np.array(self.AO.image_data[int(y_cent),:])
        y_cross_vals = np.array(self.AO.image_data[:,int(x_cent)])
        self.x_cross_section.setData(hor_vals,x_cross_vals)
        self.y_cross_section.setData(hor_vals,y_cross_vals)
        return
    
    def updateFitResults(self):
        self.fitResults_L.setText(self.AO.fit_report)
        return
    
    def set_start_CA_button(self):
        self.stop_CA_B.setEnabled(False)
        self.run_CA_B.setEnabled(True)
        return
    
    def set_stop_CA_button(self):
        self.stop_CA_B.setEnabled(True)
        self.run_CA_B.setEnabled(False)
        return
    
    def set_use_of_field_rotation(self):
        self.AO.use_field_rotation = self.use_field_rot_B.isChecked()
    
    def flip_x_action(self):
        self.AO.flip_image_x = self.flip_x_btn.isChecked()
        return

    def flip_y_action(self):
        self.AO.flip_image_y = self.flip_y_btn.isChecked()
    
    def start_control_algorithm(self):
        self.AO.controlAlgorithmIsRunning = True
        return
    
    def stop_control_algorithm(self):
        self.AO.controlAlgorithmIsRunning = False
        return    
    
    def _opt_pos_down(self):
        #Adjust optimal position one pixel down
        self.AO._y_CA -= 1
        self.update_opt_label()
        self.draw_setpoint()        
        return
    
    def _opt_pos_up(self):
        #Adjust optimal position one pixel up
        self.AO._y_CA += 1
        self.update_opt_label()
        self.draw_setpoint()        
        return
    
    def _opt_pos_left(self):
        #Adjust optimal position one pixel left
        self.AO._x_CA -= 1
        self.update_opt_label()
        self.draw_setpoint()        
        return
    
    def _opt_pos_right(self):
        #Adjust optimal position one pixel right
        self.AO._x_CA += 1
        self.update_opt_label()
        self.draw_setpoint()        
        return
    
    def extractAboutCent_action(self):
        [yc,xc] = center_of_mass(self.AO.image_data)
        #if self.AO.flip_image_x == True:
        #f self.AO.flip_image_y == True:
        #if self.AO.flip_image_x == False and self.AO.flip_image_y == False:
        #    x0,y0 = xI,yI
        
        #Keep track of where the setpoint is in full resolution. 
        
        if self.AO.flip_image_x == True:
            x0 = self.AO.camera.max_width - xc - self.ROI_size/2
            y0 = yc - self.ROI_size/2
        if self.AO.flip_image_y == True: 
            x0 = xc - self.ROI_size/2
            y0 = self.AO.camera.max_height - yc - self.ROI_size/2
        if self.AO.flip_image_x == False and self.AO.flip_image_y == False:
            x0,y0 = xc - self.ROI_size/2,yc - self.ROI_size/2
        
        x0,y0 = int(x0),int(y0)

        self.extAboutCent_B.setEnabled(False)
        self.AO.camera.stop_video_capture()
        print("Extracting ROI about (%0.0f,%0.0f)" % (x0,y0))
        self.AO.camera.set_roi(start_x=x0,
                               start_y=y0,
                               height = self.ROI_size,
                               width = self.ROI_size)
        self.AO.camera.start_video_capture()
        return

    def extractROI(self):
        '''Extract a small region around given input position'''
        self.AO.camera.stop_video_capture()
        xI,yI = int(self.x0_roi.text()),int(self.y0_roi.text())
        if self.AO.flip_image_x == True:
            x0 = self.AO.camera.max_width - xI - self.ROI_size
            y0 = yI
        if self.AO.flip_image_y == True: 
            x0 = xI
            y0 = self.AO.camera.max_height - yI - self.ROI_size
        if self.AO.flip_image_x == False and self.AO.flip_image_y == False:
            x0,y0 = xI,yI
        
        print("Extracting ROI about (%0.0f,%0.0f)" % (x0,y0))
        self.AO.camera.set_roi(start_x=x0,
                               start_y=y0,
                               height = self.ROI_size,
                               width = self.ROI_size)
        self.AO.camera.start_video_capture()
        
    def set_field_rotation_direction(self):
        self.invert_field_rot_angle = self.invert_field_rot_B.isChecked()
        return
    
    def update_field_rotation_angle(self):
        '''Loads the value in self.field_rotation_filepath 
        and converts it to an angle between 0 and 360 degrees'''
        
        #Load in the file
        try:
            init_read = np.loadtxt(self.field_rotation_filepath)
        except:
            print("\nCannot find file at %s... setting field rotation angle to 0.0 deg\n" % self.field_rotation_filepath)
            self.AO.field_rotation_ang = 0.0
            return 
        
        #Read in the value stored in the file
        init_val = np.array([init_read])[0]
        
        #Invert it if desired
        if self.invert_field_rot_angle == True:
            init_val = -1 * init_val
        
        #Input file is in centigesmal, convert to 0 -> 360 deg. 
        if init_val < 0.0 and init_val >= -180.0:
            init_val = init_val + 360
        
        #Check if value is outside bounds 
        if init_val < 0.0 or init_val > 360: 
            print("Field Rotation Value of %0.1f is outside bounds. Setting to zero" % init_val)
            self.AO.field_rotation_ang = 0.0
            return 

        #Output and store result
        print("Field Rotation Angle: %0.1f deg\n" % init_val)
        self.AO.field_rotation_ang = init_val 
        return 
    
    def get_full_resolution(self):
        self.extAboutCent_B.setEnabled(True)

        self.AO.camera.stop_video_capture()
        self.AO.camera.set_roi(start_x=0,
                               start_y=0,
                               height = self.AO.camera.max_height,
                               width = self.AO.camera.max_width)
        self.AO.camera.start_video_capture()
        
    def update_opt_label(self):
        self.current_x_opt_L.setText("{:0.0f}".format(self.AO._x_CA))
        self.current_y_opt_L.setText("{:0.0f}".format(self.AO._y_CA))
    
    #-------------AUXILLARY FUNCTION-------------
    def _update_opt_(self):
        print("Setting Optimal Position to X: {xpos} and Y: {ypos}".format(xpos=self._x_opt.text(),ypos=self._y_opt.text()))
        self.AO._x_CA,self.AO._y_CA = int(self._x_opt.text()),int(self._y_opt.text())
        self.update_opt_label()
        self.draw_setpoint()        
        return
    
    def _update_camera_settings(self):
        self.AO.camera.set_control_value(asi.ASI_GAIN,
                                         int(self.gainL.text()))
        self.AO.camera.set_control_value(asi.ASI_EXPOSURE,
                                         int(self.exposureL.text()))
        self.AO.noise_level = int(self.noiseThreshold_L.text())
        self.AO.signalThreshold = int(self.signalThreshold_L.text())
    
    def update_about_centroid_action(self):
        [yc,xc] = center_of_mass(self.AO.image_data)
        self.AO._x_CA = xc
        self.AO._y_CA = yc
        self.update_opt_label()
        self.draw_setpoint()
        print("Updating about centroid")
        return
    
    def draw_setpoint(self):
        nx,ny = np.shape(self.AO.image_data)
        h_line_x,h_line_y = np.arange(ny),np.ones(ny)*self.AO._y_CA
        v_line_x,v_line_y = np.ones(nx)*self.AO._x_CA,np.arange(nx)
        #x_circ,y_circ = self.draw_circle(x0=self.AO._x_CA,y0=self.AO._y_CA)
        self.circ_setpoint_h.setData(h_line_x,h_line_y)
        self.circ_setpoint_v.setData(v_line_x,v_line_y)

        
    def get_centroid(self):
        [yc,xc] = center_of_mass(self.AO.image_data)
        self._x_centroid.setText("{:0.0f}".format(xc))
        self._y_centroid.setText("{:0.0f}".format(yc))

        x_circ,y_circ = self.draw_circle(x0=xc,y0=yc)
        self.circ_psf.setData(x_circ,y_circ)
        return
    
    #--------------EXIT Functions------------------
    def close_application(self):
        
        self.stop_control_algorithm()
        
        try:
            self.thread.quit()
            self.thread.close()
        except:
            print("Error closing CA thread or the CA never ran")

        print("Closing Adaptive Optics GUI...")
        try:
            self.AO.ser.close()
        except:
            print("Error closing serial connection")
        
        self.close()
        
    def threaded_control_algorithm(self):
        #Create new thread so GUI does not freeze
        self.thread = QThread()
        self.thread.start()

        #Move AO object to thread
        self.AO.moveToThread(self.thread)

        #Connect signals and slots???
        #self.thread.started.connect(self.AO.run_control_algorithm)
        
        
        #self.run_CA_B.clicked.connect(self.threaded_control_algorithm)
        self.run_CA_B.clicked.connect(self.AO.run_control_algorithm)
        self.stop_CA_B.clicked.connect(lambda: self.AO.stop_control_algorithm())

        self.AO.finished.connect(self.thread.quit)
        
        #Run the thread

    def threaded_raster_scan(self):
        #Create new thread so GUI does not freeze
        self.thread2 = QThread()

        #Move AO object to thread
        self.AO.moveToThread(self.thread2)

        #Connect signals and slots???
        self.thread2.started.connect(self.AO.perform_raster_scan)
        self.AO.finished.connect(self.thread2.quit)
        
        #Run the thread
        self.thread2.start()
        
def window():
    app = QApplication(sys.argv)
    win = AO_interface()
    win.show()
    sys.exit(app.exec_())
    
window()

