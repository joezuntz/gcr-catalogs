import os
import sqlite3
import numpy as np
from GCR import BaseGenericCatalog
from .utils import md5, is_string_like

__all__ = ['DC2TruthCatalogReader', 'DC2TruthCatalogLightCurveReader']


class DC2TruthCatalogReader(BaseGenericCatalog):
    """
    DC2 truth catalog reader

    Parameters
    ----------
    filename : str
        path to the sqlite database file
    table_name : str
        table name
    is_static : bool
        whether or not this is for static objects only
    base_filters : str or list of str, optional
        set of filters to always apply to the where clause
    """

    native_filter_string_only = True

    def _subclass_init(self, **kwargs):
        self._filename = kwargs['filename']

        self._table_name = kwargs.get('table_name', 'truth')
        self._is_static = kwargs.get('is_static', True)

        base_filters = kwargs.get('base_filters')
        if base_filters:
            if is_string_like(base_filters):
                self.base_filters = (base_filters,)
            else:
                self.base_filters = tuple(base_filters)
        else:
            self.base_filters = tuple()

        if not os.path.isfile(self._filename):
            raise ValueError('{} is not a valid file'.format(self._filename))

        if kwargs.get('md5') and md5(self._filename) != kwargs['md5']:
            raise ValueError('md5 sum does not match!')

        self._conn = sqlite3.connect(self._filename)

        # get the descriptions of the columns as provided in the sqlite database
        cursor = self._conn.cursor()
        if self._is_static:
            results = cursor.execute('SELECT name, description FROM column_descriptions;')
            self._column_descriptions = dict(results.fetchall())
        else:
            self._column_descriptions = dict()

        results = cursor.execute('PRAGMA table_info({});'.format(self._table_name))
        self._native_quantity_dtypes = {t[1]: t[2] for t in results.fetchall()}

        if self._is_static:
            self._quantity_modifiers = {
                'mag_true_u': 'u',
                'mag_true_g': 'g',
                'mag_true_r': 'r',
                'mag_true_i': 'i',
                'mag_true_z': 'z',
                'mag_true_y': 'y',
                'agn': (lambda x: x.astype(np.bool)),
                'star': (lambda x: x.astype(np.bool)),
                'sprinkled': (lambda x: x.astype(np.bool)),
            }

    def _generate_native_quantity_list(self):
        return list(self._native_quantity_dtypes)

    @staticmethod
    def _obtain_native_data_dict(native_quantities_needed, native_quantity_getter):
        """
        Overloading this so that we can query the database backend
        for multiple columns at once
        """
        return native_quantity_getter(native_quantities_needed)

    def _iter_native_dataset(self, native_filters=None):
        cursor = self._conn.cursor()

        if native_filters is not None:
            all_filters = self.base_filters + tuple(native_filters)
        else:
            all_filters = self.base_filters

        if all_filters:
            query_where_clause = 'WHERE ({})'.format(') AND ('.join(all_filters))
        else:
            query_where_clause = ''

        def dc2_truth_native_quantity_getter(quantities):
            # note the API of this getter is not normal, and hence
            # we have overwritten _obtain_native_data_dict
            dtype = np.dtype([(q, self._native_quantity_dtypes[q]) for q in quantities])
            query = 'SELECT {} FROM {} {};'.format(
                ', '.join(quantities),
                self._table_name,
                query_where_clause
            )
            # may need to switch to fetchmany for larger dataset
            return np.array(cursor.execute(query).fetchall(), dtype)

        yield dc2_truth_native_quantity_getter

    def _get_quantity_info_dict(self, quantity, default=None):
        if quantity in self._column_descriptions:
            return {'description': self._column_descriptions[quantity]}
        return default


def convert_mag_to_nanoJansky(mag):
    """Convert an AB mag to nanoJansky (for a constant flux-density source).

    For a constant df/dnu source
    """
    AB_mag_zp_wrt_Jansky = 8.90  # Definition of AB
    AB_mag_zp_wrt_nanoJansky = 2.5 * 9 + AB_mag_zp_wrt_Jansky  # 9 is from nano=10**(-9)

    return 10**(-0.4 * (mag - AB_mag_zp_wrt_nanoJansky))


def convert_nanoJansky_to_mag(flux):
    """Convert a nanoJansky flux to AB mag (for a constant flux-density source).

    For a constant df/dnu source
    """
    AB_mag_zp_wrt_Jansky = 8.90  # Definition of AB
    AB_mag_zp_wrt_nanoJansky = 2.5 * 9 + AB_mag_zp_wrt_Jansky  # 9 is from nano=10**(-9)

    return -2.5 * np.log10(flux) + AB_mag_zp_wrt_nanoJansky


class DC2EmulateCatalogLightCurveReader(DC2TruthCatalogLightCurveReader)
    def _subclass_init(self, **kwargs):
        self._seed_salt = 20190228  # Phil Marshall's birthday

    # Here are the generate_modifiers entries to take truth->emulate
    @classmethod
    def emulate_uncertainty(filt, mag_true):
        """Take an input mag and filter and re-sample with Gaussian noise

        Returns
        ---
        mag, mag_err, flux, flux_err


        This is appropriate for sky-background-dominated noise
        Filter-dependent sigma re-sampling, currently hard-coded.
        """
        from numpy import random

        five_sigma_mag_floor = {'u': 27.5, 'g': 27.5, 'r': 27.5,
                                'i': 27.5, 'z': 27.5, 'y': 27.5}  # in mag
        five_sigma_flux_err = {f: convert_mag_to_nanoJansky(m)
                               for f, m in five_sigma_mag_floor.items()}
        one_sigma_flux_err = {f: err/5 for f, err in five_sigma_flux_err.items()}

        flux_true = convert_mag_to_nanoJansky(mag_true)
        flux_err = one_sigma_flux_err[filt] * np.ones_like(mag_true)
        flux_emulated = random.normal(scale=flux_err, seed=seed)
        flux_err_emulated = flux_err * np.ones_like(mag_true)

        mag_emulated = convert_nanoJansky_to_mag(flux_emulated)
        mag_err_emulated = (2.5/2) * np.log10((flux_emulated + flux_err) /
                                              (flux_emulated - flux_err))

        return mag_emulated, mag_err_emulated, flux_emulated, flux_err_emulated

    def _iter_native_dataset(self, native_filters=None):
        """Use the Truth catalog generator and then replace columns"""
        while 1:
            data = yield from super()._iter_native_dataset(native_filters=native_filters)
            # seed from salt + objectId
            seed = self._seed_salt + data['uniqueId']
            mag_emulated, mag_err_emulated, flux_emulated, flux_err_emulated =
                self.emulate_uncertainty(data['mag'], seed=seed)
            filter_name_from_num = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}
            filt = filter_name_from_num[data['filter']]
            data['mag_{}'.format(filt)] = mag_emulated
            data['magerr_{}'.format(filt)] = mag_err_emulated
            data['psFlux_{}'.format(filt)] = flux_emulated
            data['psFluxErr_{}'.format(filt)] = flux_emulated

        yield data


class DC2TruthCatalogLightCurveReader(BaseGenericCatalog):
    """
    DC2 truth catalog reader for light curves

    Parameters
    ----------
    filename : str
        path to the sqlite database file
    table_light_curves : str
        light curve table name
    table_summary : str
        summary table name
    table_obs_metadata : str
        observation metadata table name
    base_filters : str or list of str, optional
        set of filters to always apply to the where clause
    """

    native_filter_string_only = True

    def _subclass_init(self, **kwargs):
        self._filename = kwargs['filename']

        self._tables = dict()
        self._tables['light_curves'] = kwargs.get('table_light_curves', 'light_curves')
        self._tables['summary'] = kwargs.get('table_summary', 'variables_and_transients')
        self._tables['obs_meta'] = kwargs.get('table_obs_metadata', 'obs_metadata')

        base_filters = kwargs.get('base_filters')
        if base_filters:
            if is_string_like(base_filters):
                self.base_filters = (base_filters,)
            else:
                self.base_filters = tuple(base_filters)
        else:
            self.base_filters = tuple()

        if not os.path.isfile(self._filename):
            raise ValueError('{} is not a valid file'.format(self._filename))

        if kwargs.get('md5') and md5(self._filename) != kwargs['md5']:
            raise ValueError('md5 sum does not match!')

        self._conn = sqlite3.connect(self._filename)
        cursor = self._conn.cursor()
        self._dtypes = dict()
        for table, table_name in self._tables.items():
            results = cursor.execute('PRAGMA table_info({});'.format(table_name))
            self._dtypes[table] = {t[1]: t[2] for t in results.fetchall()}
        self._dtypes['light_curves'].update(self._dtypes['obs_meta'])
        del self._dtypes['obs_meta']

    def _generate_native_quantity_list(self):
        return list(self._dtypes['light_curves'])

    @staticmethod
    def _obtain_native_data_dict(native_quantities_needed, native_quantity_getter):
        """
        Overloading this so that we can query the database backend
        for multiple columns at once
        """
        return native_quantity_getter(native_quantities_needed)

    def _iter_native_dataset(self, native_filters=None):
        cursor = self._conn.cursor()

        if native_filters is not None:
            all_filters = self.base_filters + tuple(native_filters)
        else:
            all_filters = self.base_filters

        if all_filters:
            query_where_clause = 'WHERE ({})'.format(') AND ('.join(all_filters))
        else:
            query_where_clause = ''

        id_col_name = 'uniqueId'
        dtype = np.dtype([(id_col_name, self._dtypes['summary'][id_col_name])])
        query = 'SELECT DISTINCT {} FROM {} {};'.format(
            id_col_name,
            self._tables['summary'],
            query_where_clause
        )
        ids_needed = np.array(cursor.execute(query).fetchall(), dtype)[id_col_name]

        for id_this in ids_needed:
            def dc2_truth_light_curve_native_quantity_getter(quantities):
                dtype = np.dtype([(q, self._dtypes['light_curves'][q]) for q in quantities])
                query = 'SELECT {0} FROM {1} JOIN {2} ON {1}.{4}={5} AND {1}.{3}={2}.{3};'.format(
                    ', '.join(quantities),
                    self._tables['light_curves'],
                    self._tables['obs_meta'],
                    'obshistid',
                    id_col_name,
                    id_this # pylint: disable=cell-var-from-loop
                )
                return np.array(cursor.execute(query).fetchall(), dtype)
            yield dc2_truth_light_curve_native_quantity_getter
