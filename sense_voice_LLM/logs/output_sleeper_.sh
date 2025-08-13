
#!/bin/bash
# save raw env of the running container
raw_path="$PATH"
raw_ld_path="$LD_LIBRARY_PATH"
raw_py_path="$PYTHONPATH"

# import env from parent
export $(echo $PARENT_ENV|tr -s '@@' '\n' |xargs)

# source custom ENV
. ./path.sh || echo ''
# patch env by the raw env
export PATH=$PATH:$raw_path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$raw_ld_path
export PYTHONPATH=$PYTHONPATH:$raw_py_path

# create log path
log_path=""
if [ ! -z $LOG_FILE ]; then
    mkdir -p "${LOG_FILE%/*}"
    if [ -z $SLURM_ARRAY_TASK_ID ]; then
        log_path="${LOG_FILE/JOB/$((RANK+1))}"
    else
        log_path="${LOG_FILE/JOB/${SLURM_ARRAY_TASK_ID}_$((RANK+1))}"
    fi
fi

# print env && cmd to logfile
echo "################### env ###################" > $log_path
env | sed 's/^/#/g' >> $log_path
echo "################### info ###################" >> $log_path
now=`TZ='Asia/Shanghai' date -d now +'%Y-%m-%d %H:%M:%S'`
echo "data now: $now" >> $log_path
cat /etc/raw_hostname | xargs echo "run on node:" >> $log_path
echo -e "###########################################\n" >> $log_path
# run the task command
(/usr/sbin/sshd -D -p 6666) >> $log_path 2>&1
