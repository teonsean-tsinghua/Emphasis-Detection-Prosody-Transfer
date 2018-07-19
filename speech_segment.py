from py_speech_seg import speech_segmentation as seg

#seg_point = seg.multi_segmentation("audio/original/en.wav",16000,256,128,plot_seg=False,save_seg=True)
seg_point = seg.multi_segmentation("audio/original/cn.wav",16000,256,128,plot_seg=False,save_seg=True)

