from pandas import DataFrame
import numpy as np
from scipy.signal import hilbert, welch
from scipy.stats import kurtosis
from rulframework.data.feature.ABCFeatureExtractor import ABCFeatureExtractor

from scipy.signal import hilbert


class EnvelopeFeatureExtractor(ABCFeatureExtractor):

    def extract(self, raw_data: DataFrame) -> DataFrame:
        pass

    def envelope_spectrum_kurtosis(self, signal, fs):
        # 计算希尔伯特变换得到信号的包络
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)

        # 计算包络的功率谱密度
        f, Pxx = welch(amplitude_envelope, fs=fs, nperseg=1024)

        # 计算功率谱密度的峭度
        env_kurtosis = kurtosis(Pxx)

        return env_kurtosis

    def kurtosisValue(self, Signal, samplingFre):
        fre1 = 0
        fre2 = 1500

        # 希尔伯特变换
        envelopeSpectrum = np.abs(self.hilbert(Signal))

        # 幅度谱
        amplitudeSpectrum = envelopeSpectrum ** 2

        # 找出大于信号长度的最小的2的幂次方
        lenFFT = 2 ** (np.ceil(np.log2(len(amplitudeSpectrum))))

        # 去除直流分量
        amplitudeSpectrum -= np.mean(amplitudeSpectrum)

        # 对幅度谱进行快速傅里叶变换
        spectrum = np.fft.fft(amplitudeSpectrum, lenFFT)

        # 功率谱
        powerSpectrum = spectrum * np.conj(spectrum) / lenFFT

        # 求频率轴
        freAxis = samplingFre * np.arange(lenFFT / 2) / lenFFT

        # 获取指定频域上的索引幅值
        amp1 = int(round(fre1 * lenFFT / samplingFre))
        amp2 = int(round(fre2 * lenFFT / samplingFre))

        kurtValue = kurtosis(np.abs(spectrum[amp1:amp2]) * 2 / lenFFT)
        return kurtValue

    # 更新 hilbert 函数的实现，使用 scipy 库中的 hilbert 函数来计算希尔伯特变换
    @staticmethod
    def hilbert(signal):
        hilbert_transform = np.abs(hilbert(signal))
        return hilbert_transform


if __name__ == '__main__':
    fs = 1000  # 采样率
    t = np.arange(0, 1, 1 / fs)
    f1 = 50
    f2 = 120
    signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    extractor = EnvelopeFeatureExtractor()
    # 计算包络谱峭度
    kurt = extractor.envelope_spectrum_kurtosis(signal, fs)
    print("Envelope Spectrum Kurtosis:", kurt)

    # todo 傅里叶变换->频谱->希尔伯特变换->包络谱->提取峭度