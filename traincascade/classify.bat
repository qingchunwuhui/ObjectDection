opencv_traincascade.exe -data data\ -vec pos_image\pos.vec -bg neg_image\neg.dat -numPos 165 -numNeg 600 -numStages 15 -w 24 -h 24 -minHitRate 0.90 -maxFalseAlarmRate 0.5 -mode ALL
pause