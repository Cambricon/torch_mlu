This directory contains the git patches for PyTorch. TORCH_MLU strongly depends on these patches to work,
so before compiling PyTorch, these patches should be applied to PyTorch.

If you want to make patches for PyTorch, here are some tips:

* Change the files that you need in PyTorch.

* Use `git diff` command to generate the patch.

For example, after you have changed the PyTorch files, you can use the following command:

```bash
git diff > your_generated_patch.diff
```

In addition, please notice that the associated changes are suggested arranging into one patch file.

* Place your generated patch into torch_mlu/pytorch_patches folder

You can apply patches using the `scripts/apply_patches_to_pytorch.sh` in two modes:

* Apply all patches: run the script without any arguments.

* Apply specified patches: sometimes you may want to apply patches standalone for test useage, in this case, you can run the script following one or more patch files. For more information, you can run `apply_patches_to_pytorch.sh -h` for help.
