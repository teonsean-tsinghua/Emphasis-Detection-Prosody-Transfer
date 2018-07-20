# Emphasis-Detection-Prosody-Transfer
首先需要下载 https://cloud.tsinghua.edu.cn/d/7947958556a0417cbdc6/ 中的两个wav文件，放在目录audio/original下，作为数据集。  
同时，需要到 http://www.sppas.org 下载并安装sppas的依赖，将文件夹SPPAS放在项目根目录下。
## Speech Segment
此步骤将原始音轨切分成若干句音频。
### Dependencies
* python3
* ffmpeg
* tensorflow
* inaSpeechSegmenter
* pydub
### Steps
1. （optional）分别运行speech_segment.py中两段语句，在text/目录下会生成两个csv文件。该步骤耗时较长，因此csv文件将上传至repo中。
2. 运行apply_segment.py,将cn.wav和en.wav两个原始音轨中的男声片段与女声片段切分出来，分别存储在audio/segment/cn(en)/male(female)目录下。
3. （optional）运行speech2text.py，调用百度的语音识别api，将切分后的语音转录为文字，以csv格式存放在text文件夹下。四个csv文件同样上传至repo中。
