output="tmp"
input="test_512"
mode="/Users/alex/Desktop/proj/jpeg-6b-raw"
s=0
e=186
for (( i="$s"; i < "$e"; i+=1 ))
do
    for ((N=1; N <=30; N=N+1))
    do
        #echo $i-progressive 
        "$mode"/cjpeg -quality "$N" -outfile /Users/alex/Desktop/"$output"/jpg420/"$i"_"$N".jpg  /Users/alex/Desktop/"$input"/"$i".bmp
        "$mode"/djpeg -bmp -outfile /Users/alex/Desktop/"$output"/jpg420/"$i"_"$N".bmp  /Users/alex/Desktop/"$output"/jpg420/"$i"_"$N".jpg
        "$mode"/cjpeg -progressive -quality "$N" -outfile /Users/alex/Desktop/"$output"/jpgprog/"$i"_"$N".jpg  /Users/alex/Desktop/"$input"/"$i".bmp
        "$mode"/djpeg -bmp -outfile /Users/alex/Desktop/"$output"/jpgprog/"$i"_"$N".bmp  /Users/alex/Desktop/"$output"/jpgprog/"$i"_"$N".jpg
        echo $i
    done
done
        
for (( i="$s"; i < "$e"; i+=1 ))
do
    for ((N=30; N <=51; N=N+1))
    do        
        ffmpeg -i /Users/alex/Desktop/"$input"/"$i".png -c:v libx265 -x265-params crf="$N" -vf "fps=25,format=yuv444p" /Users/alex/Desktop/"$output"/hevc_444/"$i"_"$N".hevc
        ffmpeg -i /Users/alex/Desktop/"$output"/hevc_444/"$i"_"$N".hevc -vsync 0 /Users/alex/Desktop/"$output"/hevc_444/"$i"_"$N".png
        echo $i
    done
done
        
for (( i="$s"; i < "$e"; i+=1 ))
do
    for ((N=30; N <=51; N=N+1))
    do        
        ffmpeg -i /Users/alex/Desktop/"$input"/"$i".png -c:v libx265 -x265-params crf="$N" -vf "fps=25,format=yuv420p" /Users/alex/Desktop/"$output"/hevc_420/"$i"_"$N".hevc
        ffmpeg -i /Users/alex/Desktop/"$output"/hevc_420/"$i"_"$N".hevc -vsync 0 /Users/alex/Desktop/"$output"/hevc_420/"$i"_"$N".png

        echo $i
    done
done
        
for (( i="$s"; i < "$e"; i+=1 ))
do
    for ((N=30; N <=51; N=N+1))
    do        
        ffmpeg -i /Users/alex/Desktop/"$input"/"$i".png -c:v libx264 -crf "$N" -vf "fps=25,format=yuv420p" /Users/alex/Desktop/"$output"/h264_420/"$i"_"$N".mp4
        ffmpeg -i /Users/alex/Desktop/"$output"/h264_420/"$i"_"$N".mp4 -vsync 0 /Users/alex/Desktop/"$output"/h264_420/"$i"_"$N".png
        echo $i
    done
done
        
for (( i="$s"; i < "$e"; i+=1 ))
do
    for ((N=30; N <=51; N=N+1))
    do        
        ffmpeg -i /Users/alex/Desktop/"$input"/"$i".png -c:v libx264 -crf "$N" -vf "fps=25,format=yuv444p" /Users/alex/Desktop/"$output"/h264_444/"$i"_"$N".mp4
        ffmpeg -i /Users/alex/Desktop/"$output"/h264_444/"$i"_"$N".mp4 -vsync 0 /Users/alex/Desktop/"$output"/h264_444/"$i"_"$N".png

        echo $i
    done
done
        
for (( i="$s"; i < "$e"; i+=1 ))
do
    for ((N=30; N <=51; N=N+1))
    do        
        bpgenc -q "$N" -o /Users/alex/Desktop/"$output"/bpg_420/"$i"_"$N".bpg /Users/alex/Desktop/"$input"/"$i".png
        bpgdec -o /Users/alex/Desktop/"$output"/bpg_420/"$i"_"$N".png /Users/alex/Desktop/"$output"/bpg_420/"$i"_"$N".bpg
        echo $i
    done
done

for (( i="$s"; i < "$e"; i+=1 ))
do
    for ((N=30; N <=51; N=N+1))
    do        
        bpgenc -f 444 -q "$N" -o /Users/alex/Desktop/"$output"/bpg_444/"$i"_"$N".bpg /Users/alex/Desktop/"$input"/"$i".png
        bpgdec -o /Users/alex/Desktop/"$output"/bpg_444/"$i"_"$N".png /Users/alex/Desktop/"$output"/bpg_444/"$i"_"$N".bpg
        echo $i
    done
done




