'''
kalman2d - 2D Kalman filter using OpenCV

Based on http://jayrambhia.wordpress.com/2012/07/26/kalman-filter/

Copyright (C) 2014 Simon D. Levy

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
This code is distributed in the hope that it will be useful,

MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this code. If not, see <http://www.gnu.org/licenses/>.
'''

import cv

class Kalman2D(object):
    '''
    A class for 2D Kalman filtering
    '''

    def __init__(self, x=0, y=0, processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1):
        '''
        Constructs a new Kalman2D object.  
        For explanation of the error covariances see
        http://en.wikipedia.org/wiki/Kalman_filter
        '''

        self.kalman = cv.CreateKalman(4, 2, 0)
        self.kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)
        self.kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
        self.kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

        cv.SetIdentity(self.kalman.measurement_matrix)
        self.kalman.transition_matrix[0,2] = 1
        self.kalman.transition_matrix[1,3] = 1
        #self.kalman.state_pre[0, 0] = x
        #self.kalman.state_pre[1, 0] = y
        
        self.kalman.state_post[0, 0] = x
        self.kalman.state_post[1, 0] = y
        
        cv.SetIdentity(self.kalman.measurement_matrix)
        cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(processNoiseCovariance))
        cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))
        cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(errorCovariancePost))
        
        self.predicted = None
        self.corrected = None
    
    def update(self, x, y):
        '''
        Updates the filter with a new X,Y measurement
        '''
        
        self.kalman_measurement[0, 0] = x
        self.kalman_measurement[1, 0] = y
        
        self.predicted = cv.KalmanPredict(self.kalman)
        self.corrected = cv.KalmanCorrect(self.kalman, self.kalman_measurement)

    def update_none(self):
        '''
        Updates the filter with a no x,y
        '''
        self.predicted = cv.KalmanPredict(self.kalman)
        
    def getPredicted(self):
        return self.predicted[0,0], self.predicted[1,0]

    def getCorrected(self):
        return self.corrected[0,0], self.corrected[1,0]
