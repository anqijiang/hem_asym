import os
from copy import deepcopy


class _Defaults:
    _frames_per_session = 10000
    _frame_rate = 15.49
    _lap_size = 0.61
    _pico_to_cm = 200 / _lap_size
    _figure_width = 24.5
    _suite2p_path = os.path.join('suite2p', 'plane0')
    _distplot_kwargs = {'hist': True,
                        'kde': True,
                        'hist_kws': {'edgecolor': 'black'},
                        'kde_kws': {'linewidth': 3}}
    _base_path = 'G:\\Analysis\\Axon\\'
    _date_fmt = '%Y%m%d'
    _beh_fmt = '{mouse_name}_{date}_final.mat'

    def set(self, key, val):
        k = f'_{key}'
        assert hasattr(self, k), f'key {key} not found'
        setattr(self, k, val)

    @property
    def frames_per_session(self):
        return self._frames_per_session

    @property
    def frame_rate(self):
        return self._frame_rate

    @property
    def lap_size(self):
        return 0.61

    @property
    def pico_to_cm(self):
        return self._pico_to_cm

    @property
    def figure_width(self):
        return self._figure_width

    @property
    def suite2p_path(self):
        return self._suite2p_path

    @property
    def distplot_kwargs(self):
        return deepcopy(self._distplot_kwargs)

    @property
    def base_path(self):
        return self._base_path

    @property
    def date_fmt(self):
        return self._date_fmt

    @property
    def beh_fmt(self):
        return self._beh_fmt


defaults = _Defaults()
