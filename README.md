# Emphasis-Detection-Prosody-Transfer
首先需要下载 https://cloud.tsinghua.edu.cn/d/7947958556a0417cbdc6/ 中的两个wav文件，放在目录audio/original下，作为数据集。
## Speech Segment
此步骤将原始音轨切分成若干句音频。
### Dependencies
* python3
* ffmpeg
* tensorflow
* inaSpeechSegmenter
### Steps
1. 分别运行speech_segment.py中两段语句，在text/目录下会生成两个csv文件。该步骤耗时较长，因此csv文件将上传至repo中。
