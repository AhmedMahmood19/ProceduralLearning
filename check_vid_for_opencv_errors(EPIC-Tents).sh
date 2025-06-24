#!/bin/bash

# Input and output directories
directory="/workspace/Egoprocel/videos/Epic-Tents-Old"
outdirectory="/workspace/Egoprocel/videos/Epic-Tents"

# -----------------------------------------------------------------------------
# The following section should be run first to identify GoPro MP4 videos that contain
# problematic metadata (e.g., TCD, SOS) that can interfere with OpenCV processing.
# -----------------------------------------------------------------------------
for file in "$directory"/*.MP4; do
    stream_info=$(ffmpeg -i "$file" 2>&1)
    if echo "$stream_info" | grep -q "GoPro TCD" && echo "$stream_info" | grep -q "GoPro SOS"; then
        echo "$file needs to be re-encoded so OpenCV can read it"
    fi
done

# -----------------------------------------------------------------------------
# The following section should be run after you have manually created a list of files that need re-encoding
# -----------------------------------------------------------------------------
# files=(
#     "01.tent.090617.gopro.MP4"
#     "02.tent.120617.gopro.MP4"
#     # Don't need to do for 03-04
#     "05.tent.061417.gopro.MP4"
#     "06.tent.150617.gopro.MP4"
#     "07.tent.160617.gopro.MP4"
#     # Don't need to do for 08
#     "09.tent.170717.gopro.MP4"
#     "10.tent.170717.gopro.MP4"
#     "11.tent.190717.gopro.MP4"
#     "12.tent.190717.gopro.MP4"
#     "13.tent.200717.gopro.MP4"
#     "14.tent.081119.gopro.MP4"
#     "15.tent.150817.gopro.MP4"
#     "16.tent.160817.gopro.MP4"
#     "17.tent.140917.gopro.MP4"
#     # Don't need to do for 18
#     "19.tent.150917.gopro.MP4"
#     "20.tent.190917.gopro.MP4"
#     "21.tent.200917.gopro.MP4"
#     "22.tent.200917.gopro.MP4"
#     "23.tent.200917.gopro.MP4"
#     # Don't need to do for 24–28
#     "29.tent.081119.gopro.MP4"
# )

# # -----------------------------------------------------------------------------
# # Perform re-encoding on the listed files
# # -----------------------------------------------------------------------------
# for filename in "${files[@]}"; do
#     input_file="$directory/$filename"
#     output_file="$outdirectory/$filename"

#     echo "▶️ Re-encoding $input_file to $output_file"
#     ffmpeg -i "$input_file" -map 0:v:0 -map 0:a:0 -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k -movflags +faststart -y "$output_file"
# done

# echo "✅ Re-encoding complete."