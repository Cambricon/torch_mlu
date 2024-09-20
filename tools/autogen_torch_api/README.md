# Usage
```bash
cd tools/autogen_torch_api
python autogen_pytorch_apis.py main
python update_pytorch_apis.py --new "new_api_support_list.yaml" --old "current_api_support_list.yaml"
```
After executed above commands, the new api support list will be updated into `./api_support_list/torch_api.yaml`

The torchvision.ops section still need to be copied from the original api list manually.
