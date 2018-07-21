from inaSpeechSegmenter import Segmenter, seg2csv


media = 'audio/original/cn.wav'
seg = Segmenter()
segmentation = seg(media)
seg2csv(segmentation, 'csv/cn_segment.csv')
media = 'audio/original/en.wav'
seg = Segmenter()
segmentation = seg(media)
seg2csv(segmentation, 'csv/en_segment.csv')