def getmetadata(filename):
    from python_speech_features import mfcc
    import scipy.io.wavfile as wav
    import numpy as np

    (rate, sig) = wav.read(filename)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, nfft=960, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, 0)

    return feature
