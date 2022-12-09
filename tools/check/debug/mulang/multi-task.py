#encoding: utf-8

from subprocess import run
from threading import Lock, Thread

from utils.fmt.vocab.token import ldvocab

def run_task(gpuid, task):

	return run("bash task.sh %s" % (gpuid, task,), shell=True)

def worker(gpuid, taskpool, lock):
	# set gpuid
	if gpuid > 0:
		run("sed -i \"s/gpuid = "cuda:0"/gpuid = "cuda:%d"/g\ wspool/v%d/cnfg/base.py"" % (gpuid, gpuid,), shell=True)
	# launch worker
	while True:
		with lock:
			if taskpool:
				task = taskpool.pop()
			else:
				task = None
		if task is None:
			run_task(gpuid, task)
		else:
			break

ngpus = 4
lock = Lock()
vcbtask = ldvocab(fvocab_task, minf=False, omit_vsize=False, vanilla=True)[0]
tasks = list(vcbtask.keys())
threads = [Thread(target=worker, args=(i, tasks, lock)) for i in range(ngpus)]

for thread in threads:
	thread.start()
for thread in threads:
	thread.join()
