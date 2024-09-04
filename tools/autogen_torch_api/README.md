# Usage
1. python autogen_pytorch_apis.py 2.3 2>&1|tee temp.rst
2. python update_pytorch_apis.py --new temp.rst --old ../../docs/pytorch_API_support/source/API_support_list/API_support_list.rst

After executed above commands, the new api support list will be updated into temp.rst

The torchvision.ops section still need to be copied from the original api list manually.
