# Kakao-Arena
Melon Playlist Continuation

본 Git repository는 카카오 아레나의 Melon Playlist Continuation의 최종 결과 제시를 위한 공간입니다.

프로젝트를 위해 제작된 코드는 다음과 같습니다.

* mel_song_predict : mel_spetrogram에 PCA와 CNN을 적용하여 song을 예측하는 코드입니다.
* tag_predict : Denoising Auto Encoder를 활용하여 tag를 예측하는 코드입니다.
* made_result : 각각의 코드에서 예측한 플레이리스트의 song과 tag를 json으로 저장하는 코드입니다. 

### mel_song_predict를 실행하기 위해서 카카오 아레나에서 제공한 mel_spectrogram을 사용하였으나, 
### 파일의 크기가 큰 관계로 자체적으로 전처리한 데이터를 첨부하며 해당 데이터의 위치는 본 Git repository의 ___입니다.

