#!/bin/bash
#set -e

ignore_check_global=${1:-"no_ignore_check"}
is_VM=`lscpu | grep -i hypervisor`
ignore_irq=0

while getopts "i" opt
do
  case $opt in
    i)
        ignore_irq=1;;
    ?)
	echo "Unrecognized parameter."
	exit 1;;
  esac
done

function ignore_check() {
  local ignore_check=${1:-$ignore_check_global}
  if [ ${ignore_check} != "ignore_check" ]; then 
    return 1
  else
    return 0
  fi
}

function print_log() {
  echo -e "\033[31m ERROR: Run following command: cd torch_mlu/tools/perf_env_check/; sudo bash set_env.sh\033[0m"
}

OS_NAME=NULL
function checkOs() {
  echo -e "\033[1;34mChecking whether current OS is supported: \033[0m"
  if [[ -f "/etc/os-release" ]];then
    OS_NAME=$(cat /etc/os-release | awk -F '=' '{if($1=="NAME") print $2}')
    if [[ ${OS_NAME} == "Ubuntu" ]] || [[ ${OS_NAME}=="Debian"* ]]; then
      return 0
    else
      return 1
    fi
  else
    echo -e "\033[31m ERROR: Only Support Ubuntu and Debian.\033[0m"
    return 1
  fi
}

# Check whether the dataset is mounted
function check_mount() {
  dataset_path=$1
  dataset_first_path=$(echo $dataset_path | awk -F '/' '{print $2}')
  all_mount_paths=$(df -i | awk '{if(NR>1)print $6}')
  for path in $all_mount_paths
  do
    first_path=$(echo $path | awk -F '/' '{print $2}')
    if [[ $dataset_path =~ $path && $dataset_first_path == $first_path ]]
    then
      echo -e "\033[33mWARNING: The $dataset_path is mounted, it may degrade performance.\033[0m"
      return 0
    fi
  done
  echo -e "\033[32m  No mounted path found!\033[0m"
  ignore_check "no_ignore_check"
}

function check_dataset_path() {
  echo -e "\033[1;34mChecking whether the training dataset is mounted: \033[0m"
  if [ -z "$PYTORCH_TRAIN_DATASET" ]; then
    echo -e "\033[33mWARNING: \$PYTORCH_TRAIN_DATASET is not set, mount check will be skipped.\033[0m"
    return 0
  else
    check_mount $PYTORCH_TRAIN_DATASET
  fi
}

function checkCPUInfo() {
    cpu_model=$(cat /proc/cpuinfo | awk -F ':' '{if ($1 ~ "model name") print $2}' | uniq)
    cpu_physical_core_num=$(cat /proc/cpuinfo |grep "physical id"|sort|uniq | wc -l)
    processor_num=$(cat /proc/cpuinfo | grep "processor" | wc -l)
    echo -e "\033[32m$cpu_model\033[0m"
    echo -e "\033[32m CPU Physical Core Nums: $cpu_physical_core_num\033[0m"
    echo -e "\033[32m CPU Processor Nums: $processor_num\033[0m"
}

function CheckCPUExclusiveDocker() {
  top_out=$(ps aux)
  non_sys_proc=$(echo "$top_out"| awk 'NR > 1{
      pid = $2
      user = $1
      cpu_usage = $3
      command = $11

      if (user != "root" && user >= 1000 && pid >= 1000 && command !~ "^/sbin/" && command !~ "^/lib/" && command !~ "^/usr/" && command !~ "^/bin/") {
        printf("\033[1;31mThe following processes are not system processes and do not belong to the current user, please check if you need to kill them: \033[0m")
      }
  }')
  if [ -z "$non_sys_proc" ]; then
    echo -e "\033[1;32mNo non-system process found \033[0m"
    return 0
  else
    echo -e "\033[1;31mFollowing processes are determined to be non-system processes and do not belong to the current user, please check if you need to kill them:\033[0m"
    ps aux | awk ' NR > 1{
      pid = $2
      user = $1
      cpu_usage = $3
      command = $11

      if (user != "root" && user >= 1000 && pid >= 1000 && command !~ "^/sbin/" && command !~ "^/lib/" && command !~ "^/usr/" && command !~ "^/bin/") {
          printf("\033[31m  PID: %d, User: %s, CPU Usage: %s, Command: %s\n\033[0m", pid, user, cpu_usage, command)
      }
    }'
    ignore_check "ignore_check"
  fi
}

function CheckCPUExclusivePhysicalMachine () {
  active_users=$(who | awk '{print $1}')
  num_users=$(echo "$active_users" | wc -l)

  if [ "$num_users" -eq 1 ]; then
    # exclude rt user
    filtered_users=$(echo "$active_users" | grep -v "^root$")

    if [ -n "$filtered_users" ]; then
      echo "There is only one active user on the physical machine: $filtered_users"
    else
      echo "There are no non-system users on the physical machine."
    fi
  else
    echo -e "\033[31m  There are multiple active users on the physical machine, please make sure you are the only non-system active user on this machine. \033[0m"
    ignore_check
  fi

}

function CheckCPUExclusive() {
  echo -e "\033[1;34mRunning CPU exclusive check: \033[0m"
  checkCPUInfo
  if [ -f "/.dockerenv" ]; then
    echo -e "\033[1;34mDocker container env found, performing a CPU exclusive check in docker container: \033[0m"
    CheckCPUExclusiveDocker
  else
    echo -e "\033[1;34mPhysical machine env found, performing a CPU exclusive check on a physical machine: \033[0m"
    CheckCPUExclusivePhysicalMachine
  fi
}

function checkCPUPerfMode() {
  echo -e "\033[1;34mRunning CPU perf mode checks: \033[0m"
  if [ "$OS_NAME" == "Ubuntu" ]; then
    echo -e "\033[1;34mChecking linux-tools version: \033[0m"
    installed_version=$(apt list --installed | grep linux-tools-$(uname -r) | awk '{print $2}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}')
    bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]; then
      echo -e "\033[31m  ERROR: linux-tools-$(uname -r) does not match os kernel version. \033[0m"
    else
      echo -e "\033[32m  linux-tools version matches os kernel version! \033[0m"
    fi
  elif [[ ${OS_NAME}=="Debian"* ]]; then
    : 
  else
    echo -e "\033[31m  ERROR: Check CPU Performance Mode Failed. Only Support Ubuntu and Debian. \033[0m"
    ignore_check
  fi
  if [[ -e /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]];then
    performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "$performance_mode" != "performance" ]
    then
      echo -e "\033[31m  ERROR: The CPU $performance_mode Mode Enabled! \033[0m"
      print_log
      ignore_check "ignore_check"
    else
      echo -e "\033[32m  The CPU Performance Mode Enabled!\033[0m"
    fi
  else
    echo -e "\033[31mWARNING: Cannot get CPU frequency scaling info, Maybe you are running in a Virtual Machine! Please expose CPU frequency scaling capabilities to VM or make sure enable CPU performance mode on Host system!\033[0m"
    echo -e "\033[31mFor example, if you are using VMware, you can edit the VM's '.vmx' configuration file, add the following:\033[0m"
    echo -e "\033[31m	 vhv.enable = \"TRUE\"\033[0m"
    echo -e "\033[31m	 monitor_control.enable_fullcpuid = \"TRUE\"\033[0m"
    echo -e "\033[31mThen reboot the VM.\033[0m"
    return 1
  fi
}

function irqbalanceCheck() {
  echo -e "\033[1;34mRunning irqbalance checks: \033[0m"
  if [ -f "/.dockerenv" ]; then
    echo -e "\033[1;31mDocker container env found, using irqbalance inside a container is not recommanded, please running irqbalance on host system. Skipping this check...\033[0m"
    return 1
  fi
  if [[ -n "$is_VM" ]];then
    echo -e "\033[33mWARNING: You are running on a Virtual Machine! Using irqbalance may cause performance issues.\033[0m"
    echo -e "\033[33mAlternatively, you can also handle IRQs manually by setting /proc/irq/<IRQ>/smp_affinity.\033[0m"
    if [[ "$ignore_irq" -eq 1 ]];then
      echo -e "\033[32mNot using irqbalance, Ignore irqbalance check on this VM...\033[0m"
      return 0
    else
      echo -e "\033[33mIf you are not using irqbalance and want to ignore this check on VM, you can pass '-i' option when running this script.\033[0m"
      echo -e "\033[32mUsing irqbalance, Continue irqbalance check on this VM...\033[0m"
      echo -e "\033[33mWARNING: You may have shared CPUs on this VM. This script can only do limited checks. Please check CPU status on Host System whether there are other VMs sharing these vCPUs.\033[0m"
      echo -e "\033[33mYou can look for CPU Reservation(need to be 100%) and CPU Affinity(need to be assigned to specific pCPUs) in VM settings to see if the vCPUs are dedicated. You also need to check the Resource Pool Configuration to see if the VM is part of a Resource Pool which means it is shared.\033[0m"
      num_cpu=`nproc`
      if [[ $num_cpu -le 1 ]];then
        echo -e "\033[31mThere only 1 CPU on this VM. Disable irqbalance...\033[0m"
        echo -e "\033[31mPlease check CPU config on Host System and make sure the number of vCPUs pinned to this VM is greater than 1.\033[0m"
        return 1
      fi
    fi
  fi

  irqbalance_min_version="1.5.0"
  if [[ $OS_NAME=="Ubuntu" ]] || [[ ${OS_NAME}=="Debian"* ]]; then
    # check whether correct version of irqbalance is installed
    irqB_version=$(dpkg -l | grep irqbalance | awk '{print($3)}' | cut -d "-" -f 1)
    if [ ! -n "$irqB_version" ]; then
      echo -e "\033[31mERROR: irqbalance is not installed. \033[0m"
      ignore_check
    elif test "$(echo ${irqB_version} ${irqbalance_min_version} | tr " " "\n" | sort -V | head -n 1)" != "1.5.0"
    then
      echo -e "\033[31mERROR: irqbalance minimal version is 1.5.0, please upgrade irqbalance\033[0m"
      ignore_check
    else
      irqbalance_status=$(service irqbalance status)
      if [[ "$irqbalance_status" =~ "running" ]]; then
          echo -e "\033[32mIrqbalance is running... \033[0m"
      else
          echo -e "\033[31mIrqbalance is not running, please start irqbalance service using 'systemctl start irqbalance'\033[0m"
	  return 1
      fi
    fi
  else
    echo -e "\033[31mERROR: Check irqbalance status Failed. Only Support Ubuntu and Debian. \033[0m"
    ignore_check
  fi
}

function ACSCheck() {
  echo -e "\033[1;34mRunning ACS check: \033[0m"
  SCRIPT_PATH=$(dirname $(readlink -f $0))
  if [[ -f "${SCRIPT_PATH}/check_ACS_status.sh" ]]; then
    ACSStatus=$(sudo bash ${SCRIPT_PATH}/check_ACS_status.sh | while IFS= read -r line; do
      if [[ $line == *"enabled"* ]]; then
        echo -e "\033[31m  $line\033[0m"
        ignore_check
      fi
    done)
  else
    echo -e "\033[31m  check_ACS_status.sh not found, please ensure check_ACS_status.sh is in same directory as perf_env_check.sh. \033[0m"
    return 2
  fi
  echo -e "\033[32m  All MLUs' ACS are disabled! \033[0m"
}

if [ $(id -u) -eq 0 ]; then
  echo -e "\033[1;32mWith sudo privileges, a full environment check will be performed: \033[0m"
  # In this check, we only focus on PYTORCH_TRAIN_DATASET, if this env variable is not set, we consider this behavior as legal and will pass the check.
  check_dataset_path
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mMount checks passed! \033[0m"; } || { echo -e "\033[33mMount checks failed! Continue other checks...\033[0m";}
  checkOs
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mOS version check passed! \033[0m"; } || { echo -e "\033[1;31mOS version check failed! Continue other checks...\033[0m";}
  # Currently we can't strictly determine whether a process is a system process, therefore we only warn users instead of shutting down the whole script.
  CheckCPUExclusive
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mCPU exclusive checks passed! \033[0m"; } || { echo -e "\033[1;31mCPU exclusive checks failed! Continue other checks...\033[0m";}
  checkCPUPerfMode
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mCPU performance mode checks passed! \033[0m"; } || { echo -e "\033[1;31mCPU performance mode checks failed! Continue other checks...\033[0m";}
  irqbalanceCheck
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mIrqbalance checks passed! \033[0m"; } || { echo -e "\033[1;31mIrqbalance checks failed! Continue other checks...\033[0m";}
  ACSCheck
  ACS_check_status=$?
  if [[ $ACS_check_status -eq 0 ]]; then
    echo -e "\033[1;32mACS checks passed! \033[0m"
  elif [[ $ACS_check_status -eq 2 ]]; then
    echo -e "\033[1;31mACS checks skipped! \033[0m"
  else
    echo -e "\033[1;31mACS checks failed! \033[0m"
    exit 1
  fi
else
  echo -e "\033[1;32mEnv check script will be run without a sudo privilege, only limited environment checks will be performed: \033[0m"
  # In this check, we only focus on PYTORCH_TRAIN_DATASET, if this env variable is not set, we consider this behavior as legal and will pass the check.
  check_dataset_path
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mMount checks passed! \033[0m"; } || { echo -e "\033[1;31mMount checks failed! Continue other checks...\033[0m";}
  checkOs
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mOS version check passed! \033[0m"; } || { echo -e "\033[1;31mOS version check failed! Continue other checks...\033[0m";}
  CheckCPUExclusive
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mCPU exclusive checks passed! \033[0m"; } || { echo -e "\033[1;31mCPU exclusive checks failed! Continue other checks...\033[0m";}
  checkCPUPerfMode
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mCPU performance mode checks passed! \033[0m"; } || { echo -e "\033[1;31mCPU performance mode checks failed! Continue other checks...\033[0m";}
  irqbalanceCheck
  [[ $? -eq 0 ]] && { echo -e "\033[1;32mIrqbalance checks passed! \033[0m"; } || { echo -e "\033[1;31mIrqbalance checks filed! Continue other checks...\033[0m";}
fi
