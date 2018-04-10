from __future__ import division, print_function
import os
from astropy.io import fits
from GCR import BaseGenericCatalog
from skimage.transform import resize
import pandas as pd
import glob
__all__ = ['FocalPlaneCatalog']

class FitsFile(object): # from dc1.py but using hdu=0
    def __init__(self, path, rebin=0):
        self._path = path
        self._file_handle = fits.open(self._path, mode='readonly', memmap=True, lazy_load_hdus=True)
        self.data = self._file_handle[0].data #pylint: disable=E1101
        if rebin > 0:
            xdim, ydim = self.data.shape
            self.data = resize(self.data,(int(xdim/rebin),int(ydim/rebin)),preserve_range=True)
    def __del__(self):
        del self.data
        del self._file_handle[0].data #pylint: disable=E1101
        self._file_handle.close()
        del self._file_handle

class Sensor(object):
    def __init__(self, path, rebinning=16):
        self.path = path
        self.filename = os.path.split(path)[1]
        aux = self.filename.split('_')
        self.parent_visit = aux[2]
        self.parent_raft = aux[4]
        self.name = aux[5]
        self.rebinning = rebinning
    def get_data(self):
        return FitsFile(self.path,rebin=self.rebinning).data
class Raft(object):
    def __init__(self,name,visit):
        self.name = name
        self.visit = visit
        self.sensors = {}
    def add_sensor(self,sensor):
        if (sensor.parent_raft == self.name) & (sensor.parent_visit == self.visit) \
              & (sensor.name not in self.sensors.keys()):
            self.sensors.update({sensor.name:sensor})
        else:
            print('Cannot add sensor from a different raft/visit or sensor already present')
            
class FocalPlane(object):
    def __init__(self,visit):
        visit = str(visit)
        self.visit = visit
        self.rafts = {}
    def add_raft(self,raft):
        if (raft.visit == self.visit) & (raft.name not in self.rafts.keys()):
            self.rafts.update({raft.name:raft})
        else:
            print('Cannot add raft from a different visit or raft already present')
    def add_sensor(self,sensor):
        if (sensor.parent_raft not in self.rafts.keys()):
            r = Raft(sensor.parent_raft,sensor.parent_visit)
            self.add_raft(r)
        self.rafts[sensor.parent_raft].add_sensor(sensor) 

class FocalPlaneCatalog(BaseGenericCatalog):
    """
    Catalog containing information about images in a single focal plane/visit
    """

    def _subclass_init(self,
                       catalog_root_dir,
                       rebinning = 16,
                       **kwargs): #pylint: disable=W0221

        assert(os.path.isdir(catalog_root_dir)), 'Catalog directory {} does not exist'.format(catalog_root_dir)
        self._filelist = glob.glob(os.path.join(catalog_root_dir,'lsst_e*.fits*'))
        self.rebinning = rebinning
        parent_path = os.path.dirname(catalog_root_dir)
        instcat_path = glob.glob(os.path.join(parent_path,'instCat/phosim*.txt'))[0]
        self.phosim_pars = pd.read_table(instcat_path, index_col=0, header=None, sep=' ').T
        self.visit = self.phosim_pars['obshistid'].values[0]
        self.focal_plane = FocalPlane(self.visit)
        
    def _load_focal_plane(self):
        for fname in self._filelist:
            self.focal_plane.add_sensor(Sensor(fname, rebinning = self.rebinning))
            
    def _generate_native_quantity_list(self):
        native_quantity_list = {'visit'}
        self._load_focal_plane()
        for raft_name, raft in self.focal_plane.rafts.iteritems():
            native_quantity_list.update({'%s' % raft_name})
            for sensor_name in list(raft.sensors.iterkeys()):
                native_quantity_list.update({('%s-%s') % (raft_name,sensor_name)})
        return native_quantity_list 

    def _iter_native_dataset(self, native_filters=None):
        assert not native_filters, '*native_filters* is not supported'
        def _native_quantity_getter(native_quantity):
            if native_quantity == 'visit':
                self._load_focal_plane()
                return self.focal_plane
            if '-' in native_quantity:
                raft_name, sensor_name = native_quantity.split('-')
                sensor = self.focal_plane.rafts[raft_name].sensors[sensor_name]
                return sensor
            if '-' not in native_quantity:
                return self.focal_plane.rafts[native_quantity]
        yield _native_quantity_getter
       
