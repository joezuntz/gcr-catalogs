"""
CosmoDC2 galaxy catalog class.
"""
from __future__ import division
import os
import re
from itertools import product
from functools import partial
import warnings
from distutils.version import StrictVersion # pylint: disable=no-name-in-module,import-error
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
from GCR import BaseGenericCatalog
from .utils import md5, first

__all__ = ['CosmoDC2GalaxyCatalog', 'UMGalaxyCatalog']
__version__ = '1.0.0'


def _calc_weighted_size(size1, size2, lum1, lum2):
    return ((size1*lum1) + (size2*lum2)) / (lum1+lum2)


def _calc_weighted_size_minor(size1, size2, lum1, lum2, ell):
    size = _calc_weighted_size(size1, size2, lum1, lum2)
    return size * (1.0 - ell) / (1.0 + ell)


def _calc_conv(mag, shear1, shear2):
    slct = mag < 0.2
    mag_corr = np.copy(mag)
    mag_corr[slct] = 1.0 # manually changing the values for when magnification is near zero.
    conv = 1.0 - np.sqrt(1.0/mag_corr + shear1**2 + shear2**2)
    return conv


def _calc_mag(conv, shear1, shear2):
    mag = 1.0/((1.0 - conv)**2 - shear1**2 - shear2**2)
    return mag


def _calc_Rv(lum_v, lum_v_dust, lum_b, lum_b_dust):
    with np.errstate(divide='ignore', invalid='ignore'):
        v = lum_v_dust/lum_v
        b = lum_b_dust/lum_b
        bv = b/v
        Rv = np.log10(v) / np.log10(bv)
        Rv[(v == 1) & (b == 1)] = 1.0
        Rv[v == b] = np.nan
        return Rv


def _calc_Av(lum_v, lum_v_dust):
    with np.errstate(divide='ignore', invalid='ignore'):
        Av = -2.5*(np.log10(lum_v_dust/lum_v))
        Av[lum_v_dust == 0] = np.nan
        return Av


def _gen_position_angle(size_reference):
    # pylint: disable=protected-access
    size = size_reference.size
    if not hasattr(_gen_position_angle, "_pos_angle") or _gen_position_angle._pos_angle.size != size:
        _gen_position_angle._pos_angle = np.random.RandomState(123497).uniform(0, 180, size)
    return _gen_position_angle._pos_angle


def _calc_ellipticity_1(ellipticity):
    # position angle using ellipticity as reference for the size or
    # the array. The angle is converted from degrees to radians
    pos_angle = _gen_position_angle(ellipticity)*np.pi/180.0
    # use the correct conversion for ellipticity 1 from ellipticity
    # and position angle
    return ellipticity*np.cos(2.0*pos_angle)


def _calc_ellipticity_2(ellipticity):
    # position angle using ellipticity as reference for the size or
    # the array. The angle is converted from degrees to radians
    pos_angle = _gen_position_angle(ellipticity)*np.pi/180.0
    # use the correct conversion for ellipticity 2 from ellipticity
    # and position angle
    return ellipticity*np.sin(2.0*pos_angle)


def _calc_lensed_magnitude(magnitude, magnification):
    magnification[magnification == 0] = 1.0
    return magnitude -2.5*np.log10(magnification)


def _add_to_native_quantity_collector(name, obj, collector):
    if isinstance(obj, h5py.Dataset):
        collector.add(name)


class BaseCosmoDC2Catalog(BaseGenericCatalog):
    """
    BaseCosmoDC2Catalog class for catalogs that have cosmoDC-like structures.
    """

    def _subclass_init(self, catalog_root_dir, catalog_filename_template, **kwargs):
        # pylint: disable=W0221
        if not os.path.isdir(catalog_root_dir):
            raise ValueError('Catalog directory {} does not exist'.format(catalog_root_dir))

        self._healpix_files = self._get_healpix_file_list(
            catalog_root_dir,
            catalog_filename_template,
            **kwargs
        )

        cosmology = kwargs.get('cosmology', {})
        cosmo_astropy_allowed = FlatLambdaCDM.__init__.__code__.co_varnames[1:]
        cosmo_astropy = {k: v for k, v in cosmology.items() if k in cosmo_astropy_allowed}
        self.cosmology = FlatLambdaCDM(**cosmo_astropy)
        for k, v in cosmology.items():
            if k not in cosmo_astropy_allowed:
                setattr(self.cosmology, k, v)

        self.version = kwargs.get('version', '0.0.0')
        if StrictVersion(__version__) < self.version:
            raise ValueError('Reader version {} is less than config version {} for'.format(__version__, self.version))

        self.lightcone = kwargs.get('lightcone', True)
        self._md5sum = kwargs.get('md5') or {}
        self.sky_area, self._native_quantities, self._quantity_info = self._process_metadata(**kwargs)
        self._quantity_modifiers = self._generate_quantity_modifiers()
        self._native_filter_quantities = {'healpix_pixel', 'redshift_block_lower'}

    def _get_group_names(self, fh): # pylint: disable=W0613
        return ['galaxyProperties']

    def _generate_native_quantity_list(self):
        return self._native_quantities

    @staticmethod
    def _generate_quantity_modifiers():
        return {}

    @staticmethod
    def _get_healpix_file_list(catalog_root_dir, catalog_filename_template, # pylint: disable=W0613
                               zlo=None, zhi=None, healpix_pixels=None,
                               check_file_list_complete=True, **kwargs):

        healpix_files = dict()
        fname_pattern = catalog_filename_template.format(r'(\d)', r'(\d)', r'(\d+)')
        for f in sorted(os.listdir(catalog_root_dir)):
            m = re.match(fname_pattern, f)
            if m is None:
                continue

            zlo_this, zhi_this, hpx_this = tuple(map(int, m.groups()))

            # check if this file is needed
            if ((zlo is not None and zhi_this <= zlo) or
                (zhi is not None and zlo_this >= zhi) or
                (healpix_pixels is not None and hpx_this not in healpix_pixels)):
                continue

            healpix_files[(zlo_this, hpx_this)] = os.path.join(catalog_root_dir, f)

        if check_file_list_complete:
            possible_zlo = [z for z in range(3) if (zlo is None or z >= zlo) and (zhi is None or z < zhi)]
            possible_hpx = list(set(hpx for _, hpx in healpix_files)) if healpix_pixels is None else healpix_pixels
            if not all(key in healpix_files for key in product(possible_zlo, possible_hpx)):
                raise ValueError('Some catalog files are missing!')

        return healpix_files

    def _collect_native_quantities(self, fh, collect_info_dict=False):
        native_quantities = set()
        collect = partial(_add_to_native_quantity_collector, collector=native_quantities)
        group_name = first(self._get_group_names(fh))
        fh[group_name].visititems(collect)

        if collect_info_dict:
            quantity_info_dict = dict()
            modifier = lambda k, v: None if k == 'description' and v == b'None given' else v.decode()
            for quantity in native_quantities:
                quantity_info_dict[quantity] = {k: modifier(k, v) for k, v in fh[group_name][quantity].attrs.items()}
            return native_quantities, quantity_info_dict

        return native_quantities

    def _check_version(self, fh, file_name):
        # pylint: disable=E1101
        catalog_version = list()
        for version_label in ('Major', 'Minor', 'MinorMinor'):
            try:
                catalog_version.append(fh['/metaData/version' + version_label].value)
            except KeyError:
                break
        catalog_version = StrictVersion('.'.join(map(str, catalog_version or (0, 0))))
        config_version = StrictVersion(self.version)
        if config_version != catalog_version:
            raise ValueError('Catalog version {} does not match config version {} for healpix file {}'.format(catalog_version, config_version, file_name))

    def _check_md5(self, file_path):
        basename = os.path.basename(file_path)
        if basename in self._md5sum:
            if md5(file_path) != self._md5sum[basename]:
                raise ValueError('md5 sum does not match for healpix file {}'.format(basename))
        else:
            warnings.warn('No md5 sum specified in the config file for healpix file {}'.format(basename))

    def _check_cosmology(self, fh, file_name, atol):
        # pylint: disable=E1101
        for name_hdf5, name_astropy in (('H_0', 'h'), ('Omega_matter', 'Om0'), ('Omega_b', 'Ob0')):
            try:
                value_catalog = fh['metaData/{}'.format(name_hdf5)].value
            except KeyError:
                warnings.warn('missing cosmology {} in metadata for healpix file {}'.format(name_hdf5, file_name))
                continue
            value_config = getattr(self.cosmology, name_astropy).value
            if abs(value_catalog - value_config) > atol:
                raise ValueError('Mismatch in cosmological parameters ({} should be {}, not {}) for healpix file {}'.format(name_hdf5, value_config, value_catalog, file_name))

    def _process_metadata(self, ensure_quantity_consistent=False, # pylint: disable=W0613
                          check_version=True, check_md5=True,
                          check_cosmology=True, cosmology_atol=1e-4, **kwargs):
        sky_area = dict()
        native_quantities = None
        quantity_info = None

        for (_, hpx_this), file_path in self._healpix_files.items():
            if check_md5:
                self._check_md5(file_path)

            file_name = os.path.basename(file_path)
            with h5py.File(file_path, 'r') as fh:
                if check_version:
                    self._check_version(fh, file_name)

                if check_cosmology:
                    self._check_cosmology(fh, file_name, cosmology_atol)

                # get sky area
                try:
                    sky_area_this = float(fh['metaData/skyArea'].value) # pylint: disable=E1101
                except KeyError:
                    sky_area_this = np.rad2deg(np.rad2deg(4.0*np.pi/768))
                if sky_area.get(hpx_this, 0) < sky_area_this:
                    sky_area[hpx_this] = sky_area_this

                # get native quantities
                if native_quantities is None or quantity_info is None:
                    native_quantities, quantity_info = self._collect_native_quantities(fh, collect_info_dict=True)
                elif (ensure_quantity_consistent and
                        native_quantities != self._collect_native_quantities(fh)):
                    raise ValueError('native quantities are not consistent among different files')

        sky_area = sum(sky_area.values())
        return sky_area, native_quantities, quantity_info

    def _iter_native_dataset(self, native_filters=None):
        for (zlo_this, hpx_this), file_path in self._healpix_files.items():
            d = {'healpix_pixel': hpx_this, 'redshift_block_lower': zlo_this}
            if native_filters is not None and not native_filters.check_scalar(d):
                continue
            with h5py.File(file_path, 'r') as fh:
                for group in self._get_group_names(fh):
                    # pylint: disable=E1101,W0640
                    yield lambda native_quantity: fh['{}/{}'.format(group, native_quantity)].value

    def _get_quantity_info_dict(self, quantity, default=None):
        q_mod = self.get_quantity_modifier(quantity)
        if callable(q_mod) or (isinstance(q_mod, (tuple, list)) and len(q_mod) > 1 and callable(q_mod[0])):
            warnings.warn('This value is composed of a function on native quantities. So we have no idea what the units are')
            return default
        return self._quantity_info.get(q_mod or quantity, default)


class CosmoDC2GalaxyCatalog(BaseCosmoDC2Catalog):
    """
    CosmoDC2 galaxy catalog class. Uses generic quantity and filter mechanisms
    defined by BaseGenericCatalog class.
    """

    def _generate_quantity_modifiers(self):
        quantity_modifiers = {
            'galaxy_id' :    'galaxyID',
            'ra':            'ra',
            'dec':           'dec',
            'ra_true':       'ra_true',
            'dec_true':      'dec_true',
            'redshift':      'redshift',
            'redshift_true': 'redshiftHubble',
            'shear_1':       'shear1',
            'shear_2':       (np.negative, 'shear2'),
            'shear_2_treecorr': (np.negative, 'shear2'),
            'shear_2_phosim':   'shear2',
            'convergence': (
                _calc_conv,
                'magnification',
                'shear1',
                'shear2',
            ),
            'magnification': (lambda mag: np.where(mag < 0.2, 1.0, mag), 'magnification'),
            'halo_id':       'uniqueHaloID',
            'halo_mass':     'hostHaloMass',
            'is_central':    (lambda x: x.astype(np.bool), 'isCentral'),
            'stellar_mass':  'totalMassStellar',
            'stellar_mass_disk':        'diskMassStellar',
            'stellar_mass_bulge':       'spheroidMassStellar',
            'size_disk_true':           'morphology/diskMajorAxisArcsec',
            'size_bulge_true':          'morphology/spheroidMajorAxisArcsec',
            'size_minor_disk_true':     'morphology/diskMinorAxisArcsec',
            'size_minor_bulge_true':    'morphology/spheroidMinorAxisArcsec',
            'position_angle_true':      (_gen_position_angle, 'morphology/positionAngle'),
            'sersic_disk':              'morphology/diskSersicIndex',
            'sersic_bulge':             'morphology/spheroidSersicIndex',
            'ellipticity_true':         'morphology/totalEllipticity',
            'ellipticity_1_true':       (_calc_ellipticity_1, 'morphology/totalEllipticity'),
            'ellipticity_2_true':       (_calc_ellipticity_2, 'morphology/totalEllipticity'),
            'ellipticity_disk_true':    'morphology/diskEllipticity',
            'ellipticity_1_disk_true':  (_calc_ellipticity_1, 'morphology/diskEllipticity'),
            'ellipticity_2_disk_true':  (_calc_ellipticity_2, 'morphology/diskEllipticity'),
            'ellipticity_bulge_true':   'morphology/spheroidEllipticity',
            'ellipticity_1_bulge_true': (_calc_ellipticity_1, 'morphology/spheroidEllipticity'),
            'ellipticity_2_bulge_true': (_calc_ellipticity_2, 'morphology/spheroidEllipticity'),
            'size_true': (
                _calc_weighted_size,
                'morphology/diskMajorAxisArcsec',
                'morphology/spheroidMajorAxisArcsec',
                'LSST_filters/diskLuminositiesStellar:LSST_r:rest',
                'LSST_filters/spheroidLuminositiesStellar:LSST_r:rest',
            ),
            'size_minor_true': (
                _calc_weighted_size_minor,
                'morphology/diskMajorAxisArcsec',
                'morphology/spheroidMajorAxisArcsec',
                'LSST_filters/diskLuminositiesStellar:LSST_r:rest',
                'LSST_filters/spheroidLuminositiesStellar:LSST_r:rest',
                'morphology/totalEllipticity',
            ),
            'bulge_to_total_ratio_i': (
                lambda x, y: x/(x+y),
                'SDSS_filters/spheroidLuminositiesStellar:SDSS_i:observed',
                'SDSS_filters/diskLuminositiesStellar:SDSS_i:observed',
            ),
            'A_v': (
                _calc_Av,
                'otherLuminosities/totalLuminositiesStellar:V:rest',
                'otherLuminosities/totalLuminositiesStellar:V:rest:dustAtlas',
            ),
            'A_v_disk': (
                _calc_Av,
                'otherLuminosities/diskLuminositiesStellar:V:rest',
                'otherLuminosities/diskLuminositiesStellar:V:rest:dustAtlas',
            ),
            'A_v_bulge': (
                _calc_Av,
                'otherLuminosities/spheroidLuminositiesStellar:V:rest',
                'otherLuminosities/spheroidLuminositiesStellar:V:rest:dustAtlas',
            ),
            'R_v': (
                _calc_Rv,
                'otherLuminosities/totalLuminositiesStellar:V:rest',
                'otherLuminosities/totalLuminositiesStellar:V:rest:dustAtlas',
                'otherLuminosities/totalLuminositiesStellar:B:rest',
                'otherLuminosities/totalLuminositiesStellar:B:rest:dustAtlas',
            ),
            'R_v_disk': (
                _calc_Rv,
                'otherLuminosities/diskLuminositiesStellar:V:rest',
                'otherLuminosities/diskLuminositiesStellar:V:rest:dustAtlas',
                'otherLuminosities/diskLuminositiesStellar:B:rest',
                'otherLuminosities/diskLuminositiesStellar:B:rest:dustAtlas',
            ),
            'R_v_bulge': (
                _calc_Rv,
                'otherLuminosities/spheroidLuminositiesStellar:V:rest',
                'otherLuminosities/spheroidLuminositiesStellar:V:rest:dustAtlas',
                'otherLuminosities/spheroidLuminositiesStellar:B:rest',
                'otherLuminosities/spheroidLuminositiesStellar:B:rest:dustAtlas',
            ),
            'position_x': 'x',
            'position_y': 'y',
            'position_z': 'z',
            'velocity_x': 'vx',
            'velocity_y': 'vy',
            'velocity_z': 'vz',
        }

        # add magnitudes
        for band in 'ugrizyY':
            if band != 'y' and band != 'Y':
                quantity_modifiers['mag_true_{}_sdss'.format(band)] = 'SDSS_filters/magnitude:SDSS_{}:observed:dustAtlas'.format(band)
                quantity_modifiers['Mag_true_{}_sdss_z0'.format(band)] = 'SDSS_filters/magnitude:SDSS_{}:rest:dustAtlas'.format(band)
                quantity_modifiers['mag_true_{}_sdss_no_host_extinction'.format(band)] = 'SDSS_filters/magnitude:SDSS_{}:observed'.format(band)
                quantity_modifiers['Mag_true_{}_sdss_z0_no_host_extinction'.format(band)] = 'SDSS_filters/magnitude:SDSS_{}:rest'.format(band)
            quantity_modifiers['mag_true_{}_lsst'.format(band)] = 'LSST_filters/magnitude:LSST_{}:observed:dustAtlas'.format(band.lower())
            quantity_modifiers['Mag_true_{}_lsst_z0'.format(band)] = 'LSST_filters/magnitude:LSST_{}:rest:dustAtlas'.format(band.lower())
            quantity_modifiers['mag_true_{}_lsst_no_host_extinction'.format(band)] = 'LSST_filters/magnitude:LSST_{}:observed'.format(band.lower())
            quantity_modifiers['Mag_true_{}_lsst_z0_no_host_extinction'.format(band)] = 'LSST_filters/magnitude:LSST_{}:rest'.format(band.lower())

        # add lensed magnitudes
        for band in 'ugrizyY':
            if band != 'y' and band != 'Y':
                quantity_modifiers['mag_{}_sdss'.format(band)] = (_calc_lensed_magnitude, 'SDSS_filters/magnitude:SDSS_{}:observed:dustAtlas'.format(band), 'magnification',)
                quantity_modifiers['mag_{}_sdss_no_host_extinction'.format(band)] = (_calc_lensed_magnitude, 'SDSS_filters/magnitude:SDSS_{}:observed'.format(band), 'magnification',)
            quantity_modifiers['mag_{}_lsst'.format(band)] = (_calc_lensed_magnitude, 'LSST_filters/magnitude:LSST_{}:observed:dustAtlas'.format(band.lower()), 'magnification',)
            quantity_modifiers['mag_{}_lsst_no_host_extinction'.format(band)] = (_calc_lensed_magnitude, 'LSST_filters/magnitude:LSST_{}:observed'.format(band.lower()), 'magnification',)

        # add SEDs
        translate_component_name = {'total': '', 'disk': '_disk', 'spheroid': '_bulge'}
        sed_re = re.compile(r'^SEDs/([a-z]+)LuminositiesStellar:SED_(\d+)_(\d+):rest((?::dustAtlas)?)$')
        for quantity in self._native_quantities:
            m = sed_re.match(quantity)
            if m is None:
                continue
            component, start, width, dust = m.groups()
            key = 'sed_{}_{}{}{}'.format(start, width, translate_component_name[component], '' if dust else '_no_host_extinction')
            quantity_modifiers[key] = quantity

        #FIXME: remove this section when these native quantity really exist.
        self._native_quantities.difference_update(set(q for q in self._native_quantities if (
            q.startswith('emissionLines/') or q.endswith('ContinuumLuminosity')
        )))

        version = StrictVersion(self.version)
        # make quantity modifiers work in older versions
        if version < StrictVersion('1.0'):
            quantity_modifiers['halo_id'] = 'UMachineNative/halo_id'

        # make quantity modifiers work in older versions
        if version < StrictVersion('0.2'):
            quantity_modifiers['halo_id'] = 'hostHaloTag'

        return quantity_modifiers


class UMGalaxyCatalog(BaseCosmoDC2Catalog):
    """
    UM galaxy catalog class. Uses generic quantity and filter mechanisms
    defined by BaseGenericCatalog class.
    """
    def _get_group_names(self, fh):
        return [k for k in fh if k.isdigit()]

    @staticmethod
    def _generate_quantity_modifiers():
        quantity_modifiers = {
            'galaxy_id' :    'galaxy_id',
            'ra_true':       'ra',
            'dec_true':      'dec',
            'redshift_true': 'redshift',
            'halo_id':       'target_halo_id',
            'halo_mass':     'target_halo_mass',
            'stellar_mass':  'obs_sm',
            'position_x': 'x',
            'position_y': 'y',
            'position_z': 'z',
            'velocity_x': 'vx',
            'velocity_y': 'vy',
            'velocity_z': 'vz',
            'is_central': (lambda x: x == -1, 'upid'),
        }

        # add magnitudes
        for band in 'gri':
            quantity_modifiers['Mag_true_{}_sdss_z0'.format(band)] = 'restframe_extincted_sdss_abs_mag{}'.format(band)
            quantity_modifiers['Mag_true_{}_lsst_z0'.format(band)] = 'restframe_extincted_sdss_abs_mag{}'.format(band)

        return quantity_modifiers


class UMShearCatalog(UMGalaxyCatalog):
    """
    UM shear catalog class.
    """
    @staticmethod
    def _generate_quantity_modifiers():
        quantity_modifiers = {
            'ra': 'ra_lensed',
            'dec': 'dec_lensed',
            'convergence': 'conv',
            'magnification': (
                _calc_mag,
                'conv',
                'shear_1',
                'shear_2',
            ),
            'shear_2_treecorr': 'shear_2',
            'shear_2_phosim':   (np.negative, 'shear_2'),
            'shear_1':   'shear_1',
        }
        return quantity_modifiers
