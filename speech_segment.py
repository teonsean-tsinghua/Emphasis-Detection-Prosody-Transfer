from inaSpeechSegmenter import Segmenter, seg2csv
#media = 'audio/original/cn.wav'
#seg = Segmenter()
#segmentation = seg(media)
#seg2csv(segmentation, 'text/cn.csv')
media = 'audio/original/en.wav'
seg = Segmenter()
segmentation = seg(media)
seg2csv(segmentation, 'text/en.csv')