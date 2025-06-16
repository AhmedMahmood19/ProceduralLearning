#!/bin/bash

directory="/media/retrocausal-train/Extreme SSD/Egoprocel/videos/Epic-Tents"

# Loop through all .mp4 files in the directory
for file in "$directory"/*.mp4; do
    # Get stream information
    stream_info=$(ffmpeg -i "$file" 2>&1)

    # Check if there are GoPro TCD or GoPro SOS metadata streams
    if echo "$stream_info" | grep -q "GoPro TCD" && echo "$stream_info" | grep -q "GoPro SOS"; then
        echo "$file needs to be re-encoded so OpenCV can read it"
    fi
done

#### REENCODING CODE FOR THE FILES PRINTED ABOVE
#### e.g. command for re-encode:
#### ffmpeg -i /home/retrocausal-train/Desktop/Procedural-Learning/EgoProceL-egocentric-procedure-learning/pc_disassembly/Head_15.mp4 \-map 0:v:0 -map 0:a:0 -c:v copy -c:a copy -y /home/retrocausal-train/Desktop/Procedural-Learning/EgoProceL-egocentric-procedure-learning/pc_disassembly/Head_15_clean.mp4

# # Base directory path
# output_suffix="_NEW"

# # List of files to re-encode (just filenames, not full paths)
# files=(
# "01.tent.090617.gopro.MP4"
# "02.tent.120617.gopro.MP4"
# "05.tent.061417.gopro.MP4"
# "06.tent.150617.gopro.MP4"
# "07.tent.160617.gopro.MP4"
# "09.tent.170717.gopro.MP4"
# "10.tent.170717.gopro.MP4"
# "11.tent.190717.gopro.MP4"
# "12.tent.190717.gopro.MP4"
# "13.tent.200717.gopro.MP4"
# "14.tent.081119.gopro.MP4"
# "15.tent.150817.gopro.MP4"
# "16.tent.160817.gopro.MP4"
# "17.tent.140917.gopro.MP4"
# "19.tent.150917.gopro.MP4"
# "20.tent.190917.gopro.MP4"
# "21.tent.200917.gopro.MP4"
# "22.tent.200917.gopro.MP4"
# "23.tent.200917.gopro.MP4"
# "29.tent.081119.gopro.MP4"
# )

# # Loop over each file and re-encode
# for filename in "${files[@]}"; do
#     input_file="$directory/$filename"
#     output_file="${input_file%.MP4}${output_suffix}.mp4"

#     echo "Re-encoding $input_file to $output_file"
#     ffmpeg -i "$input_file" -map 0:v:0 -map 0:a:0 -c:v copy -c:a copy -y "$output_file"
# done

