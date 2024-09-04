# This script is used to check MLU ACS status, and it takes a set of devices to check whether the provided
# devices has enabled ACS. If no devices are provided, all MLU devices will be checked.
# Note that running this script requires sudo privilege.

#!/bin/bash

iommu_state="OFF"

# Check ACS
process(){
	dev=$1
	# ensure that the device can be found 
	if [ ! -e "/sys/bus/pci/devices/$dev" ]; then
		echo "Error: device $dev not found"
		exit 1
	fi

	# read ACSctl value
	acs_ctrl_reg=$(setpci -s $dev CAP_EXP+100.w)

	# get ACS status
	acs_status_byte=${acs_ctrl_reg:4:2}

	# check whether ACS is actived for current device 
	if (( 0x${acs_status_byte:1:1} & 1 )); then
		echo "ACS is enabled for device $dev"
	else
		echo "ACS is disabled for device $dev"
	fi
}

if ! sudo -n true 2>/dev/null; then
    echo "Error: This script requires sudo privileges."
    exit 1
fi

if [ -z "$1" ]; then
	#if no parameter, auto find all card BDF with domain
	raw=$(lspci -D | grep cabc)
else
	#if has parameter, find this BDF with domain
	raw=$(lspci -Ds $1)
fi

if [ -z "$raw" ]; then
	echo "Error: no EP"
	exit 1
fi

# iommu enable or not
if [ -d "/sys/kernel/iommu_groups/" ]; then
	subdirectories=$(find /sys/kernel/iommu_groups/ -maxdepth 1 -type d | wc -l)
	if [ "$subdirectories" -gt 1 ]; then
		iommu_state="ON"
	fi
fi

echo '-----------------------------------------'
echo "IOMMU is [$iommu_state]"
echo '-----------------------------------------'
echo ''

count=0
for i in $raw
do
	if [ $[count%5] -eq 0 ]; then
		process $i
		echo '----------------'
	fi
	count=$[count+1]
done
