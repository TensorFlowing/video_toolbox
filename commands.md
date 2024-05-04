# Find duration
ffmpeg -i test.MOV 2>&1 | grep "Duration"

# Find number of frames
ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 test.MOV

# Find frame rate
ffmpeg -i IMG_8691.MOV 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p" test.MOV

