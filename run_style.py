# python style.py --checkpoint [ffwd_style_network_name] --style style/wave.jpg

# # python evaluate.py --checkpoint [ffwd_style_network_name] --in-path content/tesla3.jpeg --out-path result/art1.jpg

import os
import sys

for i, argv in enumerate(sys.argv):
	if i == 0:
		continue

	picture_in = argv
	print(f"Input picture is: {picture_in}")

	picture_in_basename = os.path.basename(sys.argv[1])
	file_name = picture_in_basename.split('.')[0]
	file_extension = picture_in_basename.split('.')[1]

	checkpoints = ["wave_IN", "wave_BIN"]

	for checkpoint in checkpoints:
		picture_out = os.path.join("result", file_name + "_" + checkpoint + "." + file_extension)
		print(f" 	Start Running {checkpoint}.")
		print(f" 	Output picture will be at {picture_out}")
		if "_IN" in checkpoint:
			print(f"	python evaluate.py --checkpoint {checkpoint} --in-path {picture_in} --out-path {picture_out} --IN")
			os.system(f"python evaluate.py --checkpoint {checkpoint} --in-path {picture_in} --out-path {picture_out} --IN")
		else:
			print(f"	python evaluate.py --checkpoint {checkpoint} --in-path {picture_in} --out-path {picture_out}")
			os.system(f"python evaluate.py --checkpoint {checkpoint} --in-path {picture_in} --out-path {picture_out}")
		print(f"Finish running {checkpoint}.")
