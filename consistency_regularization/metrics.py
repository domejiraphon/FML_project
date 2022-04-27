import os 

cmd = ["python cr_test.py --model_dir use_sn --use_sn",
        "python cr_test.py --model_dir alp --alp",
        "python cr_test.py --model_dir awp --awp",
        "python cr_test.py --model_dir swa"]
for k in cmd:
  os.system(k)
