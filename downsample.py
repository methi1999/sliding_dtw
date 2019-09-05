from scipy.io import wavfile
import subprocess
import glob
import os
# Lists all the wav files
final_path = '/TIMIT_down/'
wav_files_list = glob.glob('../datasets/TIMIT/TEST/*/*/*.WAV')
cmd = "./sox {0} -r 8000 {1}"
# cmd = "./sox {0} {1} highpass 100 lowpass 3200"
# print(wav_files_list)
# # Create temporary names for the wav files to be converted. They will be renamed later on.
for f in wav_files_list:
	target_path = '/'.join(f.split('/')[:2])+final_path+'/'.join(f.split('/')[3:])
	target_path = target_path[:-3]+'wav'
	# for split in target_path.split
	print(target_path)
	subprocess.call(cmd.format(f, target_path), shell=True)
	# fileName = fileName.split('/')[-1]
	# print(fileName)
# # print(wav_prime)
# # Command strings

# mv_cmd = "mv {0} {1}"
