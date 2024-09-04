#!/bin/bash
#set -e

# This script is used to help users set env before perf test. This script requires sudo privilege.
OS_NAME=NULL
function checkOs() {
  echo -e "\033[1;34mChecking whether current OS is supported: \033[0m"
  if [[ -f "/etc/os-release" ]];then
    OS_NAME=$(cat /etc/os-release | awk -F '=' '{if($1=="NAME") print $2}')
    if [[ ${OS_NAME}=="Ubuntu" ]] || [[ ${OS_NAME}=="Debian"* ]]; then
      return 0
    else
      return 1
    fi
  else
    echo -e "\033[31m ERROR: Only Support Ubuntu and Debian.\033[0m"
    return 1
  fi
}

function setCPUPerfMode() {
  if [ "$OS_NAME" == "Ubuntu" ]
  then
    installed_version=$(dpkg -l linux-tools-$(uname -r) | grep linux-tools-$(uname -r) | awk '{print $3}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}') bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      apt-get update
      apt-get install -y linux-tools-$(uname -r)
    fi
  elif [[ ${OS_NAME}=="Debian"* ]]; then
    :
  else
    echo -e "\033[31m ERROR: Set Performance Mode Failed. Only Support Ubuntu and Debian. \033[0m"
    return 1
  fi
  performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  if [ "$performance_mode" != "performance" ]
  then
    if [ $OS_NAME=="Ubuntu" ]; then
      perf_cpu=$(cpupower -c all frequency-set -g performance)
    else
      perf_cpu=$(cpufreq-set -g performance)
    fi
    echo -e "\033[32m$perf_cpu \033[0m"
    # check performance mode
    performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "$performance_mode" == "performance" ]
    then
      echo -e "\033[32m The CPU Performance Mode Enabled!\033[0m"
      return 0
    else
      echo -e "\033[31m The CPU $performance_mode Mode Enabled! Please Check It.\033[0m"
      return 1
    fi
  else
    echo -e "\033[32m The CPU $performance_mode Mode Enabled!\033[0m"
  fi
}

# For installing version-compliant irqbalance and enabling the service
function setIrqbalanceDocker {
  irqbalance_min_version="1.5.0"
  if [[ $OS_NAME=="Ubuntu" ]] || [[ $OS_VERSION=="Debian"* ]]; then
    # install irqbalance
    irqB_version=$(dpkg -l | grep irqbalance | awk '{print($3)}' | cut -d "-" -f 1)
    if [ -z "${irqB_version}" ] && [ "$(echo "${irqB_version} ${irqbalance_min_version}" | tr " " "\n" | sort -V | head -n 1)" != "1.5.0" ]; then
      echo -e "\033[32m irqbalance is not installed or irqbalance version is too low, installing latest irqbalance. \033[0m"
      apt-get update
      apt-get install irqbalance
    fi
    # If current irqbalance status is not running, start irqbalance service
    irqbalance_status=$(service irqbalance status)
    if [[ "$irqbalance_status" =~ "running" ]]; then
      echo -e "\033[32m irqbalance is already running \033[0m"
      return 0
    else
      echo -e "\033[irqbalance is not running, starting irqbalance service\033[0m"
      service irqbalabce start
      if [[ "$irqbalance_status" =~ "running" ]]; then
        echo -e "\033[32m irqbalance is already running \033[0m"
        return 0
     else
	echo -e "\033[31m failed to start irqbalance, please start irqbalance manually. \033[0m"
	return 1
      fi
    fi
  else
    echo -e "\033[31m ERROR: Start irqbalabce Failed. Only Support Ubuntu and Debian. \033[0m"
    return 1
  fi
}

# This function requires sudo, and is used to disable ACS (Access Control Services)
function disableACS {
  out=$(sudo lspci -nn | grep "cabc" | awk '{print $1}')
  echo "$out" | while IFS= read -r line; do
    dev=$line  
    if [ -z "$dev" ]; then
      echo "\033[31m Error: no device specified \033[0m"
      return 1
    fi
    if [ ! -e "/sys/bus/pci/devices/$dev" ]; then
      dev="0000:$dev"
    fi
    if [ ! -e "/sys/bus/pci/devices/$dev" ]; then
      echo "\033[31m Error: device $dev not found \033[0m"
      return 1
    fi
  
    port=$(basename $(dirname $(readlink "/sys/bus/pci/devices/$dev")))
  
    if [ ! -e "/sys/bus/pci/devices/$port" ]; then
      echo "\033[31m Error: device $port not found \033[0m"
      return 1
    fi
  while [ -e "/sys/bus/pci/devices/$port" ]
    do
      echo "device $port be found"
      #acs=$(setpci -s $port ECAP_ACS+6.w)
      sleep 1
      #setpci -s $port ECAP_ACS+6.w=$(printf "%04x" $(("0x$acs" | 0x40)))
      setpci -s $port ECAP_ACS+6.w=0x0
      port=$(basename $(dirname $(readlink "/sys/bus/pci/devices/$port")))
    done 
  done
  return 0
}

if ! sudo -n true 2>/dev/null; then
    echo "\033[1;31mError: This script requires sudo privileges. \033[0m"
    exit 1
fi

checkOs
[[ $? -eq 0 ]] && { echo -e "\033[1;32mOS version check passed! \033[0m"; } || { echo -e "\033[1;31mOS version check failed! \033[0m"; exit;}
setCPUPerfMode
[[ $? -eq 0 ]] && { echo -e "\033[1;32mSet CPU perf mode succeeded! \033[0m"; } || { echo -e "\033[1;31mSet CPU perf mode failed! \033[0m"; exit;}
setIrqbalanceDocker
[[ $? -eq 0 ]] && { echo -e "\033[1;32mStart Irqbalance succeeded! \033[0m"; } || { echo -e "\033[1;31mStart Irqbalance failed! \033[0m"; exit;}
disableACS
[[ $? -eq 0 ]] && { echo -e "\033[1;32mACS successfully disabled! \033[0m"; } || { echo -e "\033[1;31mFailed to disable ACS! please disable ACS manually. \033[0m"; exit;}
