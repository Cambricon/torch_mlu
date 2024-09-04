import sys
import os

try:
    import yapf
    from yapf.yapflib.yapf_api import FormatFile
except:
    os.system("pip install yapf")
    import yapf
    from yapf.yapflib.yapf_api import FormatFile


def main():
    filename = sys.argv[1]
    FormatFile(filename, in_place=True)
    print("writing: " + filename)


if __name__ == "__main__":
    main()
