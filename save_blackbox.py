import sys
import torchvision


def main():
    img = torchvision.io.read_image(path=sys.argv[1], mode=torchvision.io.ImageReadMode.RGB)
    img[:, 16:72, 1257:1505] = 0
    img[:, 451:604, 1257:1505] = 0
    torchvision.utils.save_image(img, path=sys.argv[2])


if __name__ == '__main__':
    main()
