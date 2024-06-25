import pyhrv, biosppy
import neurokit2 as nk
import numpy as np

def biosppy_hr(rpeaks, sampling_rate):
    sampling_rate = float(sampling_rate)
    hr_idx, hr = biosppy.signals.tools.get_heart_rate(
    beats = rpeaks, 
    sampling_rate = sampling_rate, 
    smooth = True, 
    size = 3
    )
    return {'hr' : hr}


def pyhrv_hr(rpeaks, sampling_rate):
    # https://github.com/PGomes92/pyhrv/blob/73ea446ecf7d3416fef86fc4025ddcc242398e3f/pyhrv/time_domain.py#L311
    rpeaks = correct_rpeaks_pyhrv(rpeaks, sampling_rate)
    nn = pyhrv.utils.check_input(nni = None, rpeaks = rpeaks)
    hr = pyhrv.tools.heart_rate(nn)
    return {'hr' : hr.mean()}

def pyhrv_rmssd(rpeaks, sampling_rate):
    # https://github.com/PGomes92/pyhrv/blob/73ea446ecf7d3416fef86fc4025ddcc242398e3f/pyhrv/time_domain.py#L311
    rpeaks =correct_rpeaks_pyhrv(rpeaks, sampling_rate)
    nn = pyhrv.utils.check_input(nni = None, rpeaks = rpeaks)
    nnd = pyhrv.tools.nni_diff(nn)
    rmssd_ = np.sum([x**2 for x in nnd])
    rmssd = np.sqrt(1. / nnd.size * rmssd_)

    return {'rmssd' : rmssd}

def pyhrv_sdnn(rpeaks, sampling_rate):
    # https://github.com/PGomes92/pyhrv/blob/73ea446ecf7d3416fef86fc4025ddcc242398e3f/pyhrv/time_domain.py#L173
    rpeaks =correct_rpeaks_pyhrv(rpeaks, sampling_rate)
    nn = pyhrv.utils.check_input(nni = None, rpeaks = rpeaks)
    sdnn = pyhrv.utils.std(nn)
    return {'sdnn' : sdnn}

def pyhrv_time(rpeaks, sampling_rate):
    result = dict()

    result.update({
        'pyhrv hr' : pyhrv_hr(rpeaks, sampling_rate)['hr']
    })
    result.update({
        'pyhrv rmssd' : pyhrv_rmssd(rpeaks, sampling_rate)['rmssd']
    })
    result.update({
        'pyhrv sdnn' : pyhrv_sdnn(rpeaks, sampling_rate)['sdnn']
    })
    return result


def nk_time(rpeaks, sampling_rate):
    # just for test
    out = nk.hrv_time(rpeaks, sampling_rate, False)
    return {
        'nk rmssd' : float(out['HRV_RMSSD']),
        'nk sdnn' : float(out['HRV_SDNN'])
    }
    
def nk_hr(rpeaks, sampling_rate):
    return nk.signal_rate(rpeaks, sampling_rate).mean()
    
def time(rpeaks, sampling_rate):
    result = dict()
    result.update(pyhrv_time(rpeaks, sampling_rate))
    result.update(nk_time(rpeaks, sampling_rate))
    result.update({
        'biosppy hr' : biosppy_hr(rpeaks, sampling_rate)['hr'].mean()
    })
    result['nk hr'] = nk_hr(rpeaks, sampling_rate)
    return result


import numpy as np
import biosppy
import neurokit2 as nk

'''
Note that, R-peaks alone have no information about the sampling_rate
'''

def biosppy_filter(signal, sampling_rate):
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    filtered, _, _ = biosppy.signals.tools.filter_signal(
        signal = signal, 
        ftype = 'FIR', 
        band = 'bandpass', 
        order = int(0.3 * sampling_rate), 
        frequency = [3, 45], 
        sampling_rate = sampling_rate)
    return {'filtered' : filtered}


def biosppy_rpeaks(filtered, sampling_rate):
    sampling_rate = float(sampling_rate)
    (rpeaks,) = biosppy.signals.ecg.hamilton_segmenter(
        signal = filtered, 
        sampling_rate = sampling_rate
        )

    (rpeaks,) = biosppy.signals.ecg.correct_rpeaks(
        signal = filtered, 
        rpeaks = rpeaks, 
        sampling_rate = sampling_rate, 
        tol=0.05
        )

    templates, rpeaks = biosppy.signals.ecg.extract_heartbeats(
        signal = filtered,
        rpeaks = rpeaks,
        sampling_rate = sampling_rate,
        before = 0.2,
        after= 0.4,
        )
    return {'rpeaks' : rpeaks}


def biosppy_processing(signal, sampling_rate):
    filtered = biosppy_filter(signal, sampling_rate)['filtered']
    rpeaks = biosppy_rpeaks(filtered, sampling_rate)['rpeaks']
    return {'filtered' : filtered, 'rpeaks' : np.array(rpeaks)}


def nk_filter(signal, sampling_rate):
    signal = nk.signal_sanitize(signal)
    filtered = nk.ecg_clean(
        ecg_signal = signal,
        sampling_rate = sampling_rate,
        method = 'neurokit'
    )
    return {'filtered' : filtered}

def nk_rpeaks(filtered, sampling_rate):
    temp, info = nk.ecg_peaks(
        filtered,
        sampling_rate
    )
    rpeaks = temp[temp['ECG_R_Peaks'] == 1].index
    rpeaks = list(rpeaks)
    return {'rpeaks' : rpeaks}

def nk_processing(signal, sampling_rate):
    filtered = nk_filter(signal, sampling_rate)['filtered']
    rpeaks = nk_rpeaks(signal, sampling_rate)['rpeaks']
    return {'filtered' : filtered, 'rpeaks' : np.array(rpeaks)}
    
    
def processing_aggregate(signal, sampling_rate):
    nk_result = nk_processing(signal, sampling_rate)
    biosppy_result = biosppy_processing(signal, sampling_rate)
    if nk_result['rpeaks'][0] - biosppy_result['rpeaks'][0] > sampling_rate // 4 :
        rpeaks = [biosppy_result['rpeaks'][0]] + list(nk_result['rpeaks'])
    else: rpeaks = nk_result['rpeaks']
    return {
        'rpeaks' : np.array(rpeaks),
        'nk filtered' : nk_result['filtered'],
        'biosppy filtered' : biosppy_result['filtered'],
    }
    








def correct_rpeaks_pyhrv(rpeaks, sampling_rate):
    '''
    most of functions in pyhrv is coded based on 1000 Hz, and there are few function that have sampling_rate option.
    to this end, adjust rpeaks based on sampling_rate/1000 is best option.
    note that rpeaks is array consist of indices of r peaks.
    thus, if the signal sampled at 2000 Hz, the indices are doubled.
    '''
    rpeaks = np.array(rpeaks)
    ratio = sampling_rate / 1000
    corrected = rpeaks / ratio
    return corrected


def pyhrv_nonlinear(rpeaks, sampling_rate, show = False):
    rpeaks = correct_rpeaks_pyhrv(rpeaks, sampling_rate)
    result = pyhrv.nonlinear.poincare(rpeaks = rpeaks, show = False, mode = 'dev' if show == False else 'normal')
    result = dict(result)
    result['sd1/sd2'] = result['sd1'] / result['sd2']
    return dict(result)

def nk_nonlinear(rpeaks, sampling_rate):
    output = nk.hrv_nonlinear(rpeaks, sampling_rate)
    result = dict()
    for c in output.columns:
        result[c] = output[c].item()
    return dict(result)

def nonlinear(rpeaks, sampling_rate):
    result_dict = dict()
    result_dict['pyhrv nonlinear'] = pyhrv_nonlinear(rpeaks, sampling_rate)
    result_dict['nk nonlinear'] = nk_nonlinear(rpeaks, sampling_rate)
    return result_dict




import pyhrv
import neurokit2 as nk
import numpy as np


def pyhrv_welch(rpeaks, sampling_rate, show = False):
    rpeaks = correct_rpeaks_pyhrv(rpeaks, sampling_rate)
    result = pyhrv.frequency_domain.welch_psd(rpeaks = rpeaks, show = False, mode = 'dev' if show == False else 'normal')
    return dict(result[0])

def pyhrv_ar(rpeaks, sampling_rate, show = False):
    rpeaks = correct_rpeaks_pyhrv(rpeaks, sampling_rate)
    result = pyhrv.frequency_domain.ar_psd(rpeaks = rpeaks, show = False, mode = 'dev' if show == False else 'normal')
    result = dict(result[0])
    result['ar_ratio'] = 1/result['ar_ratio']
    return result


def pyhrv_frequency(rpeaks, sampling_rate, show = False):
    result = dict()
    result.update({
        'pyhrv welch' : pyhrv_welch(rpeaks, sampling_rate, show),
        'pyhrv ar' : pyhrv_ar(rpeaks, sampling_rate, show),
    })
    return result


def nk_welch(rpeaks, sampling_rate):
    output = nk.hrv_frequency(peaks = rpeaks, sampling_rate = sampling_rate, psd_method = 'welch')
    result = dict()
    for c in output.columns:
        result[c] = output[c].item()
    return result

def nk_fft(rpeaks, sampling_rate):
    output = nk.hrv_frequency(peaks = rpeaks, sampling_rate = sampling_rate, psd_method = 'fft')
    result = dict()
    for c in output.columns:
        result[c] = output[c].item()
    return result

def nk_frequency(rpeaks, sampling_rate):
    result = dict()
    result.update({
        'nk welch' : nk_welch(rpeaks, sampling_rate),
        'nk fft' : nk_fft(rpeaks, sampling_rate),
    })
    return result
    
def frequency(rpeaks, sampling_rate):
    result = dict()
    result.update(pyhrv_frequency(rpeaks, sampling_rate))
    result.update(nk_frequency(rpeaks, sampling_rate))
    return result










import numpy as np
import biosppy
import neurokit2 as nk

'''
Note that, R-peaks alone have no information about the sampling_rate
'''

def biosppy_filter(signal, sampling_rate):
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    filtered, _, _ = biosppy.signals.tools.filter_signal(
        signal = signal, 
        ftype = 'FIR', 
        band = 'bandpass', 
        order = int(0.3 * sampling_rate), 
        frequency = [3, 45], 
        sampling_rate = sampling_rate)
    return {'filtered' : filtered}


def biosppy_rpeaks(filtered, sampling_rate):
    sampling_rate = float(sampling_rate)
    (rpeaks,) = biosppy.signals.ecg.hamilton_segmenter(
        signal = filtered, 
        sampling_rate = sampling_rate
        )

    (rpeaks,) = biosppy.signals.ecg.correct_rpeaks(
        signal = filtered, 
        rpeaks = rpeaks, 
        sampling_rate = sampling_rate, 
        tol=0.05
        )

    templates, rpeaks = biosppy.signals.ecg.extract_heartbeats(
        signal = filtered,
        rpeaks = rpeaks,
        sampling_rate = sampling_rate,
        before = 0.2,
        after= 0.4,
        )
    return {'rpeaks' : rpeaks}


def biosppy_processing(signal, sampling_rate):
    filtered = biosppy_filter(signal, sampling_rate)['filtered']
    rpeaks = biosppy_rpeaks(filtered, sampling_rate)['rpeaks']
    return {'filtered' : filtered, 'rpeaks' : np.array(rpeaks)}


def nk_filter(signal, sampling_rate):
    signal = nk.signal_sanitize(signal)
    filtered = nk.ecg_clean(
        ecg_signal = signal,
        sampling_rate = sampling_rate,
        method = 'neurokit'
    )
    return {'filtered' : filtered}

def nk_rpeaks(filtered, sampling_rate):
    temp, info = nk.ecg_peaks(
        filtered,
        sampling_rate
    )
    rpeaks = temp[temp['ECG_R_Peaks'] == 1].index
    rpeaks = list(rpeaks)
    return {'rpeaks' : rpeaks}

def nk_processing(signal, sampling_rate):
    filtered = nk_filter(signal, sampling_rate)['filtered']
    rpeaks = nk_rpeaks(signal, sampling_rate)['rpeaks']
    return {'filtered' : filtered, 'rpeaks' : np.array(rpeaks)}
    
    
def processing_aggregate(signal, sampling_rate):
    nk_result = nk_processing(signal, sampling_rate)
    biosppy_result = biosppy_processing(signal, sampling_rate)
    if nk_result['rpeaks'][0] - biosppy_result['rpeaks'][0] > sampling_rate // 4 :
        rpeaks = [biosppy_result['rpeaks'][0]] + list(nk_result['rpeaks'])
    else: rpeaks = nk_result['rpeaks']
    return {
        'rpeaks' : np.array(rpeaks),
        'nk filtered' : nk_result['filtered'],
        'biosppy filtered' : biosppy_result['filtered'],
    }
    
    































