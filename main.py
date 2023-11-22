from image import Image
from color import Color

def main():
    width = 3
    height = 2
    im = Image(width, height)
    red = Color(1,0,0)
    green = Color(0,1,0)
    blue = Color(0,0,1)
    white = Color(1,1,1)

    im.set_pixel(0,0,red)
    im.set_pixel(1,0,white)
    im.set_pixel(2,0,blue)

    im.set_pixel(0,1,red)
    im.set_pixel(1,1,white)
    im.set_pixel(2,1,blue)

    with open("test.ppm", "w") as img_file:
        im.write_ppm(img_file)

if __name__ == "__main__":
    main()